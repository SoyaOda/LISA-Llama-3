#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
import os
import time
import deepspeed
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig, MllamaForConditionalGeneration
from .segment_anything import sam_model_registry
from .segment_anything.utils.transforms import ResizeLongestSide
import cv2


class LISAForCausalLM(nn.Module):
    """
    LISAモデル：Llama 3.2 VisionとSAM (Segment Anything Model)を統合したモデル
    """

    def __init__(
        self,
        model_path="meta-llama/Llama-3.2-11B-Vision-Instruct",
        sam_checkpoint=None,
        seg_token="<seg>",
        device=None,
        max_batch_size=1,
        use_deepspeed=False,
        ds_config=None,
        local_rank=-1
    ):
        """
        LISAモデルを初期化

        Args:
            model_path (str): Llama 3.2 Vision Instructモデルのパス
            sam_checkpoint (str): SAMチェックポイントのパス
            seg_token (str): セグメンテーショントークン
            device (str): 使用するデバイス ('cuda' または 'cpu')
            max_batch_size (int): 最大バッチサイズ
            use_deepspeed (bool): DeepSpeedを使用するかどうか
            ds_config (dict): DeepSpeedの設定
            local_rank (int): 分散学習でのローカルランク（DeepSpeed用）
        """
        super().__init__()

        # DeepSpeed関連の設定を保存
        self.use_deepspeed = use_deepspeed
        self.ds_config = ds_config
        self.local_rank = local_rank

        # デバイスを設定
        if device is None:
            # CUDAが利用可能な場合はそれを使用
            if torch.cuda.is_available():
                self.device = "cuda"
            # MPSサポートを安全に確認（Apple Siliconの場合）
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                try:
                    self.device = "mps"
                    print("MPS (Metal Performance Shaders) を使用します")
                except Exception as e:
                    print(f"MPSの初期化中にエラーが発生しました: {e}")
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # DeepSpeedを使用する場合、ローカルランクがマイナスでなければローカルランクを出力
        if self.use_deepspeed and self.local_rank >= 0:
            print(f"DeepSpeedを使用します。ローカルランク: {self.local_rank}")

        print(f"使用デバイス: {self.device}")

        # セグメンテーショントークンを設定
        self.seg_token = seg_token

        # モデルの精度を設定
        self.dtype = torch.float32
        if torch.cuda.is_available():
            # 最適なデータ型を選択
            self.dtype = torch.float16  # 必要に応じてbfloat16など他の型に変更可能
        print(f"モデルの精度: {self.dtype}")

        # 各コンポーネントの初期化
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.sam = None
        self.seg_token_idx = None
        self.sam_transform = None

        # プロセッサとモデルをロード
        self.initialize_llama(model_path)

        # セグメンテーション投影レイヤー
        self.initialize_lisa_modules(sam_checkpoint)

    def initialize_llama(self, model_path):
        """
        Llama 3.2 Vision モデルを初期化
        """
        print(f"Llama 3.2 Visionモデルをロード: {model_path}")

        try:
            # 設定をロード
            config = AutoConfig.from_pretrained(model_path)
            print(f"モデル設定: {config.__class__.__name__}")

            # プロセッサをロード
            self.processor = AutoProcessor.from_pretrained(model_path)
            print(f"プロセッサがロードされました: {self.processor.__class__.__name__}")

            # モデルをロード - Llama 3.2 Vision専用のクラスを使用
            # DeepSpeed使用時と非使用時で分岐
            if self.use_deepspeed:
                # DeepSpeedを使用する場合の設定
                print("DeepSpeedを使用してモデルをロードします")

                # デバイスマップは使用しない（DeepSpeedが管理するため）
                load_params = {
                    "torch_dtype": self.dtype,
                    "trust_remote_code": True
                }

                # モデルをロード
                model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    **load_params
                )

                # DeepSpeedエンジンを初期化
                ds_engine_params = {
                    "model": model,
                    "config_params": self.ds_config,
                }

                # ローカルランクが指定されている場合は追加
                if self.local_rank >= 0:
                    ds_engine_params["config_params"]["local_rank"] = self.local_rank

                # DeepSpeedエンジンを初期化
                self.model, _, _, _ = deepspeed.initialize(**ds_engine_params)

                print("DeepSpeedエンジンが初期化されました")
            else:
                # 通常のロード（DeepSpeedなし）
                device_param = "auto" if self.device is None else self.device

                # トーチ2.0以降でのMPSサポートチェック
                load_params = {
                    "torch_dtype": self.dtype,
                    "trust_remote_code": True
                }

                # MPSまたはCUDAサポートチェック
                if device_param != "cpu":
                    try:
                        load_params["device_map"] = device_param
                    except Exception as e:
                        print(f"デバイスマップの設定中にエラー: {e}")
                        print("CPUにフォールバックします")
                        device_param = "cpu"

                self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    **load_params
                )

            self.model.eval()
            print(f"モデルがロードされました: {self.model.__class__.__name__}")

            # トークナイザーにセグメンテーショントークンを追加（存在しない場合）
            if self.seg_token not in self.processor.tokenizer.get_vocab():
                print(f"トークナイザーに {self.seg_token} トークンを追加します")
                self.processor.tokenizer.add_tokens([self.seg_token])

                # モデルのエンベディング層を拡張
                # より安全なオプションを設定してトークンを拡張
                try:
                    print("embedding層を拡張中...")
                    # 大きいモデルでのメモリ不足に対応
                    # トークナイザーサイズを保存
                    vocab_size = len(self.processor.tokenizer)

                    # 低メモリモードでトークンを追加
                    if self.device == "cpu":
                        print("低メモリモードでembedding層を拡張します")

                        # トークン情報を保存
                        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(
                            self.seg_token)
                        print(f"セグメンテーショントークンのインデックス: {self.seg_token_idx}")

                        # CPUではembedding層拡張をスキップ - tokenizerと互換性を維持するために
                        # トークンインデックスだけ取得（そしてハンドリング時に特別処理）
                        print("CPUではembedding層の拡張をスキップします")
                    else:
                        # DeepSpeed使用時は特別な処理が必要
                        if self.use_deepspeed:
                            # DeepSpeedモデルのためのトークン拡張（代替手法）
                            print("DeepSpeedモデルのための代替埋め込み拡張手法を使用します")

                            # トークンIDだけを取得し、埋め込み拡張はスキップ
                            # これはDeepSpeed環境での埋め込み変更が複雑なため
                            print("DeepSpeed環境では埋め込み拡張をスキップします")
                            self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(
                                self.seg_token)
                            print(
                                f"セグメンテーショントークンのインデックス: {self.seg_token_idx}")
                            print("トークンは追加されましたが、埋め込み拡張はスキップされました")

                            # 注意：実際の運用では、埋め込み拡張を含むモデルを事前に保存し、
                            # それをDeepSpeedでロードすることを推奨します
                        else:
                            # 通常の場合
                            self.model.resize_token_embeddings(vocab_size)

                            self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(
                                self.seg_token)
                            print(
                                f"セグメンテーショントークンのインデックス: {self.seg_token_idx}")
                            print("embedding層の拡張が完了しました")
                except Exception as e:
                    print(f"embeddings拡張中にエラー: {e}")
                    print("別の代替手段を試行...")
                    try:
                        # 最低限の対応として、トークンIDの取得のみ行う
                        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(
                            self.seg_token)
                        print(
                            f"トークン拡張はスキップしますが、セグメンテーショントークンIDを保存: {self.seg_token_idx}")
                    except Exception as e2:
                        print(f"トークンID取得中にもエラー発生: {e2}")
                        # どうしても失敗した場合は仮のトークンIDを設定
                        print("警告: 仮のトークンIDを使用します")
                        self.seg_token_idx = -1

            # セグメンテーショントークンのインデックスを取得
            self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(
                self.seg_token)
            print(f"セグメンテーショントークンID: {self.seg_token_idx}")

        except Exception as e:
            print(f"モデルのロード中にエラーが発生: {e}")
            raise

    def initialize_lisa_modules(self, sam_checkpoint=None):
        """
        LISAの追加モジュール（SAMなど）を初期化
        """
        if sam_checkpoint:
            print(f"SAMチェックポイント: {sam_checkpoint}")
            try:
                print(f"SAMチェックポイントを読み込みます: {sam_checkpoint}")
                self.sam = build_sam_vit_h(sam_checkpoint)

                # SAMを同じデバイスに移動
                print(f"SAMモデルを {self.device} デバイスに移動します...")
                self.sam.to(self.device)

                # 高速化のためにSAMモデルを半精度に変換
                if self.dtype == torch.float16 or self.dtype == torch.bfloat16:
                    print(f"SAMモデルを半精度（{self.dtype}）に変換します...")
                    self.sam = self.sam.to(self.dtype)
                    print("SAMモデルを半精度に変換しました")

                # SAMパラメータを凍結
                print("SAMのパラメータを凍結します...")
                for param in self.sam.parameters():
                    param.requires_grad = False

                print("SAMモデルが正常に初期化されました")

                # テキスト->SAMプロンプト投影（Llama 3.2 Visionの隠れ状態次元から256次元へ）
                # Llama 3.2 Visionのhidden_sizeはtext_configのhidden_sizeから取得
                try:
                    # DeepSpeedを使用している場合
                    if self.use_deepspeed:
                        # DeepSpeedモデルのconfigアクセス方法
                        if hasattr(self.model, "module"):
                            if hasattr(self.model.module, "config"):
                                # configオブジェクトを取得
                                config = self.model.module.config

                                # Mllamaモデルの場合の特殊な対応
                                if hasattr(config, "text_config"):
                                    hidden_size = config.text_config.hidden_size
                                    print(
                                        f"text_configからhidden_size取得: {hidden_size}")
                                else:
                                    # dictの場合やtext_configが存在しない場合の対応
                                    if isinstance(config, dict) and "text_config" in config:
                                        hidden_size = config["text_config"]["hidden_size"]
                                        print(
                                            f"text_config dictからhidden_size取得: {hidden_size}")
                                    else:
                                        # フォールバック: 一般的なサイズを使用
                                        print(
                                            "警告: モデル設定からhidden_sizeを取得できません。デフォルト値4096を使用します。")
                                        hidden_size = 4096  # Llama-3.2の一般的なサイズ
                            else:
                                print(
                                    "警告: モデルmoduleにconfigがありません。デフォルト値4096を使用します。")
                                hidden_size = 4096
                        else:
                            print("警告: モデルにmodule属性がありません。デフォルト値4096を使用します。")
                            hidden_size = 4096
                    else:
                        # 通常のモデル（非DeepSpeed）
                        hidden_size = self.model.config.text_config.hidden_size
                except Exception as e:
                    print(f"hidden_size取得中にエラー発生: {e}")
                    print("デフォルト値4096を使用します")
                    hidden_size = 4096  # フォールバック値

                # セグメンテーション投影を初期化
                print(f"セグメンテーション投影を初期化: {hidden_size} -> 256")
                self.seg_projection = nn.Linear(hidden_size, 256)

                # セグメンテーション投影もGPUに移動
                self.seg_projection.to(self.device, self.dtype)

                # 重みを初期化（Kaiming初期化）
                nn.init.kaiming_normal_(
                    self.seg_projection.weight, nonlinearity='relu')

            except Exception as e:
                print(f"SAMモデルの初期化中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                self.sam = None

    def transform_image(self):
        """
        SAMモデル用の画像変換オブジェクトを作成します。

        Returns:
            transform: ResizeLongestSideオブジェクト
        """
        # SAM用のリサイズ変換を作成
        transform = ResizeLongestSide(target_length=1024)
        return transform

    def preprocess_sam_image(self, image):
        """
        SAMモデル用に画像を前処理する関数

        Args:
            image: 入力画像（PIL.Image）

        Returns:
            image_embedding: SAMの画像埋め込み
        """
        try:
            print("SAM用に画像を前処理中...")

            # PIL.Image形式の画像を処理
            if isinstance(image, Image.Image):
                # PIL画像をResizeNetの入力サイズにリサイズ
                target_size = (1024, 1024)  # SAMのデフォルト入力サイズ
                original_size = image.size  # (width, height)
                original_size = (
                    original_size[0], original_size[1])  # 明示的にタプルに変換

                print(f"オリジナル画像サイズ: {original_size}")

                # PIL画像をRGBに変換してからnumpy配列に変換
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # numpy配列に変換し、[H, W, C]形式にする
                input_image = np.array(image)
                print(
                    f"入力画像のシェイプ: {input_image.shape}, 型: {input_image.dtype}")

                # 画像を[0, 1]の範囲に正規化
                input_image = input_image.astype(np.float32) / 255.0

                # [H, W, C] -> [C, H, W]に変換（PyTorch形式）
                input_image = np.transpose(input_image, (2, 0, 1))

                # バッチ次元を追加して[1, C, H, W]形式に
                input_image_torch = torch.from_numpy(input_image).unsqueeze(0)

                # Float16に変換（SAMモデルがhalf precisionの場合）
                # モデルのパラメータが半精度(float16)かを確認
                sam_param = next(self.sam.parameters())
                if sam_param.dtype == torch.float16:
                    print("入力画像を半精度(float16)に変換します")
                    input_image_torch = input_image_torch.half()

                print(f"SAM入力画像のシェイプ: {input_image_torch.shape}")
                print(f"SAM入力画像のデータ型: {input_image_torch.dtype}")

                # 重要: 入力画像をSAMモデルと同じデバイスに移動
                # SAMモデルのデバイスを取得
                sam_device = next(self.sam.parameters()).device
                print(f"SAMモデルのデバイス: {sam_device}")

                # 画像をSAMモデルと同じデバイスに移動（DeepSpeedの場合でも）
                input_image_torch = input_image_torch.to(sam_device)
                print(f"SAM入力画像のデバイス: {input_image_torch.device}")

                # SAMのimage_encoderを使用して画像埋め込みを作成
                with torch.no_grad():
                    # 画像埋め込みを計算
                    image_embedding = self.sam.image_encoder(input_image_torch)
                    print(f"画像埋め込みのシェイプ: {image_embedding.shape}")

                return image_embedding
            else:
                raise ValueError("入力画像はPIL.Image形式である必要があります")
        except Exception as e:
            print(f"SAM画像前処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def generate_segmentation(self, image, prompt, max_new_tokens=1024, top_p=0.95, temperature=0.1, top_k=None, num_beams=1, repetition_penalty=1.0, do_sample=True):
        """
        画像とプロンプトからセグメンテーションと説明テキストを生成します。

        Args:
            image: 入力画像
            prompt: 指示用プロンプトテキスト
            max_new_tokens: 最大生成トークン数
            top_p, temperature, top_k, num_beams, repetition_penalty, do_sample: 生成パラメータ

        Returns:
            {"text": 生成テキスト, "masks": 生成されたマスクのリスト}
        """
        # マスクのリストを初期化
        masks = []

        def check_image(img):
            """画像が正しい形式かチェックし、必要なら変換する"""
            from PIL import Image
            import numpy as np
            
            if isinstance(img, str):
                # 画像パスの場合はロード
                return Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                # NumPy配列の場合はPILに変換
                return Image.fromarray(img).convert('RGB')
            elif isinstance(img, Image.Image):
                # すでにPIL画像の場合はそのまま返す
                return img.convert('RGB')
            else:
                raise ValueError(f"非対応の画像形式: {type(img)}")

        try:
            # モデルのデバイスを取得（DeepSpeed使用時も同様）
            model_device = next(self.model.parameters()).device
            print(f"モデルのデバイス: {model_device}")

            # SAMモデルの画像エンコーダを使用して画像埋め込みを取得
            # SAM用の画像埋め込みを事前に計算
            sam_image_embedding = self.preprocess_sam_image(image)
            print(f"SAM画像埋め込み: 形状={sam_image_embedding.shape}, デバイス={sam_image_embedding.device}")

            # 画像が正しい形式かチェック
            pil_image = check_image(image)
            
            # Llama 3.2 Visionのメッセージ形式
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            print(f"メッセージ構造: {messages}")
            
            # チャットテンプレートの適用
            input_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors=None
            )
            
            print(f"入力テキスト: {input_text}")
            
            try:
                # 画像とテキストを一度に処理してモデル入力を作成
                print("画像とテキストを処理します...")
                model_inputs = self.processor(
                    images=pil_image,
                    text=input_text,
                    return_tensors="pt"
                )
                
                # デバッグ用に処理前の形状を出力
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"処理後の {k}: 形状={v.shape}, デバイス={v.device}, 型={v.dtype}")
                
                # すべての入力テンソルをモデルのデバイスに移動
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to(model_device)
                
                # pixel_valuesの形状を正確に確認し修正
                print(f"修正前のpixel_values形状: {model_inputs['pixel_values'].shape}")
                
                # Llama 3.2 Visionは6次元のpixel_values形状を期待:
                # [batch_size, num_concurrent_media, num_tiles, num_channels, height, width]
                if len(model_inputs['pixel_values'].shape) == 5:
                    # 5次元から6次元に変換 - num_concurrent_mediaの次元を追加
                    # 例: [B, P, C, H, W] -> [B, 1, P, C, H, W]
                    model_inputs['pixel_values'] = model_inputs['pixel_values'].unsqueeze(1)
                    print(f"5次元から6次元に変更後のpixel_values形状: {model_inputs['pixel_values'].shape}")
                elif len(model_inputs['pixel_values'].shape) > 6:
                    # 次元が多すぎる場合は適切に調整
                    print(f"異常な次元数のpixel_values: {model_inputs['pixel_values'].shape}")
                    # 必要に応じて他の変換を追加
                
                # aspect_ratio_idsを確認・修正
                if 'aspect_ratio_ids' not in model_inputs:
                    print("aspect_ratio_idsを作成します")
                    # num_concurrent_mediaの次元に合わせて作成
                    batch_size = model_inputs['pixel_values'].shape[0]
                    num_concurrent_media = model_inputs['pixel_values'].shape[1]
                    model_inputs['aspect_ratio_ids'] = torch.zeros(
                        (batch_size, num_concurrent_media),
                        dtype=torch.long, device=model_device
                    )
                
                # aspect_ratio_maskの確認・修正
                if 'aspect_ratio_mask' not in model_inputs:
                    print("aspect_ratio_maskを作成します")
                    batch_size = model_inputs['pixel_values'].shape[0]
                    num_concurrent_media = model_inputs['pixel_values'].shape[1]
                    model_inputs['aspect_ratio_mask'] = torch.ones(
                        (batch_size, num_concurrent_media),
                        dtype=torch.bool, device=model_device
                    )
                
                # 最終確認: すべてのテンソルの形状とデバイスを出力
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"最終的なモデル入力 {k}: 形状={v.shape}, デバイス={v.device}")
                
                # テキスト生成
                try:
                    # DeepSpeed環境の場合
                    if hasattr(self.model, 'module') and hasattr(self.model.module, 'generate'):
                        print("DeepSpeed環境でgenerateを実行します")
                        try:
                            generate_outputs = self.model.module.generate(
                                input_ids=model_inputs['input_ids'],
                                attention_mask=model_inputs['attention_mask'],
                                pixel_values=model_inputs['pixel_values'],
                                aspect_ratio_ids=model_inputs['aspect_ratio_ids'] if 'aspect_ratio_ids' in model_inputs else None,
                                aspect_ratio_mask=model_inputs['aspect_ratio_mask'] if 'aspect_ratio_mask' in model_inputs else None,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=50 if top_k is None else top_k,
                                num_beams=num_beams,
                                repetition_penalty=repetition_penalty,
                                output_hidden_states=True,
                                return_dict_in_generate=True
                            )
                        except Exception as deepspeed_error:
                            print(f"DeepSpeed generate中にエラーが発生しました: {deepspeed_error}")
                            # フォールバック: 非DeepSpeed方式で試す
                            print("フォールバック: 非DeepSpeed方式でgenerateを実行します")
                            generate_outputs = self.model.generate(
                                input_ids=model_inputs['input_ids'],
                                attention_mask=model_inputs['attention_mask'],
                                pixel_values=model_inputs['pixel_values'],
                                aspect_ratio_ids=model_inputs['aspect_ratio_ids'] if 'aspect_ratio_ids' in model_inputs else None,
                                aspect_ratio_mask=model_inputs['aspect_ratio_mask'] if 'aspect_ratio_mask' in model_inputs else None,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=50 if top_k is None else top_k,
                                num_beams=num_beams,
                                repetition_penalty=repetition_penalty,
                                output_hidden_states=True,
                                return_dict_in_generate=True
                            )
                    else:
                        # 通常のgenerateメソッド
                        print("通常環境でgenerateを実行します")
                        generate_outputs = self.model.generate(
                            input_ids=model_inputs['input_ids'],
                            attention_mask=model_inputs['attention_mask'],
                            pixel_values=model_inputs['pixel_values'],
                            aspect_ratio_ids=model_inputs['aspect_ratio_ids'] if 'aspect_ratio_ids' in model_inputs else None,
                            aspect_ratio_mask=model_inputs['aspect_ratio_mask'] if 'aspect_ratio_mask' in model_inputs else None,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=50 if top_k is None else top_k,
                            num_beams=num_beams,
                            repetition_penalty=repetition_penalty,
                            output_hidden_states=True,
                            return_dict_in_generate=True
                        )

                    # 生成されたトークンIDを取得
                    generate_ids = generate_outputs.sequences
                    
                    # トークンをデコード
                    decoded_text = self.processor.tokenizer.batch_decode(
                        generate_ids, skip_special_tokens=False
                    )[0]
                    
                    print(f"生成された生のテキスト: {decoded_text}")

                    # <seg>トークンの検出と処理
                    seg_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                        self.seg_token)
                    print(f"<seg>トークンID: {seg_token_id}")

                    # 生成されたトークンから<seg>トークンの位置を見つける
                    seg_positions = []
                    for batch_idx in range(generate_ids.shape[0]):
                        positions = torch.where(generate_ids[batch_idx] == seg_token_id)[0]
                        for pos in positions:
                            seg_positions.append((batch_idx, pos.item()))
                    
                    print(f"<seg>トークン位置: {seg_positions}")

                    # 各<seg>トークンについてマスクを生成
                    for batch_idx, pos in seg_positions:
                        try:
                            # この位置までのシーケンスを抽出
                            input_ids_segment = generate_ids[batch_idx, :pos+1].unsqueeze(0)
                            attention_mask_segment = torch.ones_like(input_ids_segment)

                            # フォワードパスを実行して隠れ状態を取得
                            with torch.no_grad():
                                # pixel_valuesやaspect_ratio関連のテンソルが正しい形状であることを確認
                                # batch_size=1に制限して取得
                                pixel_values_segment = model_inputs['pixel_values'][0:1]
                                aspect_ratio_ids_segment = model_inputs['aspect_ratio_ids'][0:1] if 'aspect_ratio_ids' in model_inputs else None
                                aspect_ratio_mask_segment = model_inputs['aspect_ratio_mask'][0:1] if 'aspect_ratio_mask' in model_inputs else None
                                
                                print(f"フォワードパス用 pixel_values: 形状={pixel_values_segment.shape}")
                                if aspect_ratio_ids_segment is not None:
                                    print(f"フォワードパス用 aspect_ratio_ids: 形状={aspect_ratio_ids_segment.shape}")
                                if aspect_ratio_mask_segment is not None:
                                    print(f"フォワードパス用 aspect_ratio_mask: 形状={aspect_ratio_mask_segment.shape}")
                                
                                outputs = self.model(
                                    input_ids=input_ids_segment,
                                    attention_mask=attention_mask_segment,
                                    pixel_values=pixel_values_segment,
                                    aspect_ratio_ids=aspect_ratio_ids_segment,
                                    aspect_ratio_mask=aspect_ratio_mask_segment,
                                    output_hidden_states=True,
                                    return_dict=True
                                )

                                # 最後の位置（<seg>トークン）の隠れ状態を取得
                                hidden_states = outputs.hidden_states[-1][0, -1]
                                
                                # 隠れ状態をSAMの次元（256次元）に投影
                                point_embedding = self.seg_projection(hidden_states)
                                sam_device = next(self.sam.parameters()).device
                                point_embedding = point_embedding.to(sam_device)
                                
                                print(f"SAMプロンプト埋め込み: 形状={point_embedding.shape}, デバイス={point_embedding.device}")
                                
                                # SAMデコーダを実行してマスクを生成
                                sparse_embeddings = point_embedding.unsqueeze(0)
                                mask_predictions, _ = self.sam.mask_decoder(
                                    image_embeddings=sam_image_embedding.to(sam_device),
                                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse_embeddings,
                                    dense_prompt_embeddings=None,
                                    multimask_output=False,
                                )
                                
                                # マスク予測を取得して後処理
                                mask_pred = mask_predictions[0]  # [1, 1, H, W]
                                mask_pred = torch.sigmoid(mask_pred)
                                mask_binary = (mask_pred > 0.5).float()
                                mask_np = mask_binary[0, 0].cpu().numpy()
                                masks.append(mask_np)
                                
                                print(f"マスク {len(masks)} を生成しました: 形状={mask_np.shape}")
                        
                        except Exception as e:
                            print(f"マスク生成中にエラーが発生しました: {str(e)}")
                            import traceback
                            traceback.print_exc()
                
                except Exception as e:
                    print(f"トークン生成中にエラーが発生しました: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {"masks": [], "text": f"エラー: {str(e)}"}
            
            except Exception as e:
                print(f"画像とテキストの処理中にエラーが発生しました: {str(e)}")
                import traceback
                traceback.print_exc()
                return {"masks": [], "text": f"エラー: {str(e)}"}
        
        except Exception as e:
            print(f"セグメンテーション生成中にエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"masks": [], "text": f"エラー: {str(e)}"}
        
        # 特殊トークンを含まないクリーンなテキストを取得
        clean_text = self.processor.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True
        )[0]

        # 生成テキストとマスクを含む辞書を返す
        return {"masks": masks, "text": clean_text}

    @classmethod
    def from_pretrained(cls, model_path, sam_checkpoint=None, use_deepspeed=False, ds_config=None, local_rank=-1, **kwargs):
        """
        事前学習済みモデルからLISAモデルを作成

        Args:
            model_path: モデルのパス
            sam_checkpoint: SAMチェックポイントのパス
            use_deepspeed: DeepSpeedを使用するかどうか
            ds_config: DeepSpeedの設定
            local_rank: ローカルランク（DeepSpeed用）
            **kwargs: その他の引数
        """
        return cls(
            model_path=model_path,
            sam_checkpoint=sam_checkpoint,
            use_deepspeed=use_deepspeed,
            ds_config=ds_config,
            local_rank=local_rank,
            **kwargs
        )


def build_sam_vit_h(checkpoint=None):
    """
    SAM ViT-Hモデルを構築
    """
    try:
        from .segment_anything import sam_model_registry
        print(f"SAMモデルを構築します: vit_h")
        return sam_model_registry["vit_h"](checkpoint=checkpoint)
    except Exception as e:
        print(f"SAMモデル構築中にエラー: {e}")
        return None
