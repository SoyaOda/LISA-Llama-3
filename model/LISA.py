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
                # NumPy配列の場合はPIL画像に変換
                if img.ndim == 3 and img.shape[2] == 3:
                    return Image.fromarray(img)
                else:
                    raise ValueError("NumPy画像は形状(H, W, 3)である必要があります")
            elif isinstance(img, Image.Image):
                # すでにPIL画像の場合はそのまま使用
                if img.mode != 'RGB':
                    return img.convert('RGB')
                return img
            else:
                raise ValueError("サポートされていない画像形式です")

        try:
            # モデルのデバイスを取得
            device = next(self.model.parameters()).device
            print(f"モデルのデバイス: {device}")
            
            # SAMで使用する画像処理
            try:
                # 画像が有効な形式かチェック
                image_pil = check_image(image)
                
                # SAMモデル用の画像埋め込みを生成
                sam_image_embedding = self.preprocess_sam_image(image_pil)
                print(f"SAM画像埋め込み: 形状={sam_image_embedding.shape}, デバイス={sam_image_embedding.device}")
                
                # プロセッサを使ってLlama用の画像とテキスト処理
                # Llama 3.2 Visionモデルは[image, text]の順にメッセージ形式を期待
            messages = [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
                
                # メッセージからチャットテンプレートを適用
                input_text = self.processor.apply_chat_template(
                messages,
                    add_generation_prompt=True,
                    return_tensors=None  # テキストを返す
                )
                print(f"チャットテンプレート適用後: {input_text[:100]}...")
                
                # 画像とテキストを統合して処理
                model_inputs = self.processor(
                    image_pil, 
                input_text,
                return_tensors="pt"
            )

                # テンソルをモデルのデバイスに移動
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to(device)
                
                # 入力テンソルの形状確認
                print(f"pixel_values: 形状={model_inputs['pixel_values'].shape}")
                
                # 一部のデバイスやモデル構成では6次元を期待する場合がある
                if 'pixel_values' in model_inputs and len(model_inputs['pixel_values'].shape) == 5:
                    # [batch_size, num_tiles, channels, height, width] -> [batch_size, 1, num_tiles, channels, height, width]
                    model_inputs['pixel_values'] = model_inputs['pixel_values'].unsqueeze(1)
                    print(f"pixel_values (リシェイプ後): 形状={model_inputs['pixel_values'].shape}")
                
                # aspect_ratio_ids と aspect_ratio_mask が存在しない場合は作成
                if 'aspect_ratio_ids' not in model_inputs and 'pixel_values' in model_inputs:
                    # デフォルト: 1.0のアスペクト比でタイル全体使用とマーク
                    b, n_media, n_tiles = model_inputs['pixel_values'].shape[:3]
                    model_inputs['aspect_ratio_ids'] = torch.zeros((b, n_media, n_tiles), dtype=torch.long, device=device)
                    model_inputs['aspect_ratio_mask'] = torch.ones((b, n_media, n_tiles), dtype=torch.long, device=device)
                    print("aspect_ratio_ids と aspect_ratio_mask を生成しました")
                
                # 生成パラメータ設定
                # top_kがNoneまたは大きすぎる場合の安全対策
                if top_k is None or top_k > 50:
                    top_k = 50  # 安全な最大値に制限
                
                # 異常値回避のためのパラメータ検証
                temperature = max(0.1, min(2.0, temperature))  # 0.1~2.0の範囲に制限
                top_p = max(0.1, min(0.99, top_p))  # 0.1~0.99の範囲に制限
                
                generation_params = {
                    "input_ids": model_inputs["input_ids"],
                    "attention_mask": model_inputs["attention_mask"],
                    "pixel_values": model_inputs["pixel_values"],
                    "aspect_ratio_ids": model_inputs["aspect_ratio_ids"],
                    "aspect_ratio_mask": model_inputs["aspect_ratio_mask"],
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "num_beams": num_beams,
                    "use_cache": True,
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                }
                
                # メモリ使用量を表示
                if torch.cuda.is_available():
                    print(f"生成前のGPUメモリ: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
                
                try:
                    print("DeepSpeed環境でgenerateを実行します")
                    
                    # パラメータ詳細のログ
                    print(f"生成パラメータ: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}, top_p={top_p}")
                    
                    # DeepSpeedのモジュールを使用して生成
                    generate_outputs = self.model.module.generate(**generation_params)
                except Exception as e:
                    print(f"通常のgenerateでエラーが発生しました: {str(e)}")
                    
                    try:
                        print("フォールバック: シンプルな生成方法を試行します")
                        
                        # CPUに移動して再試行する前にGPUキャッシュをクリア
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            print("GPUキャッシュをクリアしました")
                        
                        # top_kをさらに小さくして再試行
                        top_k = min(10, top_k)  # さらに小さいtop_kを使用
                        temperature = max(0.2, temperature)  # 小さすぎるtemperatureを避ける
                        do_sample = True  # サンプリングを有効に
                        
                        # CPUで処理する前に入力をCPUに移動
                        cpu_inputs = {}
                        for k, v in generation_params.items():
                            if isinstance(v, torch.Tensor):
                                cpu_inputs[k] = v.detach().cpu()
                            else:
                                cpu_inputs[k] = v
                        
                        # top_kを更新
                        cpu_inputs["top_k"] = top_k
                        cpu_inputs["temperature"] = temperature
                        cpu_inputs["do_sample"] = do_sample
                        
                        # メモリ使用量削減のためデバイスをCPUに
                        with torch.no_grad():
                            model_cpu = self.model.module.to('cpu')
                            generate_outputs = model_cpu.generate(**cpu_inputs)
                            # 結果を取得後、モデルをGPUに戻す
                            self.model.module.to(device)
                            
                    except Exception as e2:
                        print(f"フォールバック生成でもエラーが発生しました: {str(e2)}")
                        
                        # さらなるフォールバック：完全なグリーディ生成を試行
                        try:
                            print("最終フォールバック: グリーディ生成を試行します")
                            
                            # greedy search、パラメータ最小化
                            minimal_params = {
                                "input_ids": cpu_inputs["input_ids"] if 'cpu_inputs' in locals() else generation_params["input_ids"].detach().cpu(),
                                "max_new_tokens": min(128, max_new_tokens),  # トークン数制限
                                "do_sample": False,  # greedy search
                                "use_cache": True,
                                "pad_token_id": self.processor.tokenizer.pad_token_id,
                                "eos_token_id": self.processor.tokenizer.eos_token_id,
                            }
                            
                            # CPU上で実行
                            with torch.no_grad():
                                model_cpu = self.model.module.to('cpu')
                                generate_outputs = model_cpu.generate(**minimal_params)
                                # 結果を取得後、モデルをGPUに戻す
                                self.model.module.to(device)
                                
                        except Exception as e3:
                            print(f"全ての生成方法が失敗しました: {str(e3)}")
                            # ダミーのテキスト生成結果を返す
                            return {"masks": [], "text": f"テキスト生成に失敗しました: {str(e3)}"}
                
                # 生成されたトークンIDを取得
                generate_ids = generate_outputs.detach()
                
                # 生成テキストをデコード
                decoded_text = self.processor.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=False
                )[0]
                
                print(f"生成テキスト: {decoded_text[:100]}...")
                
                # トークナイザからセグメンテーショントークンのIDを取得
                seg_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
                
                if seg_token_id == self.processor.tokenizer.unk_token_id:
                    print(f"警告: セグメンテーショントークン {self.seg_token} がボキャブラリに見つかりません。UNKトークンとして処理されます。")
                
                print(f"<seg>トークンID: {seg_token_id}")
                
                # 生成テキスト内の<seg>トークン位置を検索
                try:
                    # セグメンテーショントークンの位置を見つける
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
                                pixel_values_segment = model_inputs['pixel_values'][:1]
                                aspect_ratio_ids_segment = model_inputs['aspect_ratio_ids'][:1] if 'aspect_ratio_ids' in model_inputs else None
                                aspect_ratio_mask_segment = model_inputs['aspect_ratio_mask'][:1] if 'aspect_ratio_mask' in model_inputs else None
                                
                                print(f"フォワードパス用 pixel_values: 形状={pixel_values_segment.shape}")
                                if aspect_ratio_ids_segment is not None:
                                    print(f"フォワードパス用 aspect_ratio_ids: 形状={aspect_ratio_ids_segment.shape}")
                                if aspect_ratio_mask_segment is not None:
                                    print(f"フォワードパス用 aspect_ratio_mask: 形状={aspect_ratio_mask_segment.shape}")
                                
                                # 値がNaNまたはInfでないことを確認
                                for tensor_name, tensor in [("pixel_values", pixel_values_segment), 
                                                        ("aspect_ratio_ids", aspect_ratio_ids_segment), 
                                                        ("aspect_ratio_mask", aspect_ratio_mask_segment)]:
                                    if tensor is not None and torch.isnan(tensor).any() or torch.isinf(tensor).any():
                                        print(f"警告: {tensor_name}にNaNまたはInf値が含まれています")
                                        if tensor_name == "pixel_values":
                                            # NaNやInfを0に置き換え
                                            pixel_values_segment = torch.nan_to_num(pixel_values_segment, nan=0.0, posinf=1.0, neginf=0.0)
                                            
                                try:
                                    # まずGPUで試す
                                    outputs = self.model(
                                        input_ids=input_ids_segment,
                                        attention_mask=attention_mask_segment,
                                        pixel_values=pixel_values_segment,
                                        aspect_ratio_ids=aspect_ratio_ids_segment,
                                        aspect_ratio_mask=aspect_ratio_mask_segment,
                                        output_hidden_states=True,
                                        return_dict=True
                                    )
                                except Exception as e:
                                    print(f"GPUでの隠れ状態取得中にエラー: {str(e)}")
                                    print("CPUでの処理に切り替えます")
                                    
                                    # CPUに移動して再試行
                                    try:
                                        # モデルをCPUに移動
                                        model_cpu = self.model.to('cpu')
                                        
                                        outputs = model_cpu(
                                            input_ids=input_ids_segment.to('cpu'),
                                            attention_mask=attention_mask_segment.to('cpu'),
                                            pixel_values=pixel_values_segment.to('cpu'),
                                            aspect_ratio_ids=aspect_ratio_ids_segment.to('cpu') if aspect_ratio_ids_segment is not None else None,
                                            aspect_ratio_mask=aspect_ratio_mask_segment.to('cpu') if aspect_ratio_mask_segment is not None else None,
                                            output_hidden_states=True,
                                            return_dict=True
                                        )
                                        
                                        # 処理後、モデルをGPUに戻す
                                        self.model.to(device)
                                    except Exception as cpu_error:
                                        print(f"CPUでの処理も失敗しました: {str(cpu_error)}")
                                        # ダミーの隠れ状態を作成し続行を試みる
                                        hidden_dim = self.model.module.config.text_config.hidden_size
                                        dummy_hidden_states = [[torch.zeros((1, 1, hidden_dim), device='cpu')]]
                                        
                                        outputs = type('DummyOutputs', (), {})
                                        outputs.hidden_states = dummy_hidden_states
                                        print(f"ダミーの隠れ状態を生成しました: 形状={dummy_hidden_states[-1][0].shape}")
                                        
                                # ここでoutputsを使用
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                                    # 最後の位置（<seg>トークン）の隠れ状態を取得
                                    hidden_states = outputs.hidden_states[-1][0, -1]
                                    print(f"隠れ状態: 形状={hidden_states.shape}, デバイス={hidden_states.device}")
                                    
                                    # 隠れ状態をSAMの次元（256次元）に投影
                                    point_embedding = self.seg_projection(hidden_states)
                                    print(f"投影前のpoint_embedding: 形状={point_embedding.shape}, デバイス={point_embedding.device}")
                                    
                                    # SAMデバイスを確認し、必要に応じて移動
                                    sam_device = next(self.sam.parameters()).device
                                    point_embedding = point_embedding.to(sam_device)
                                    
                                    print(f"SAMプロンプト埋め込み: 形状={point_embedding.shape}, デバイス={point_embedding.device}")
                                    
                                    # SAMデコーダを実行してマスクを生成
                                    sparse_embeddings = point_embedding.unsqueeze(0)
                                    
                                    try:
                                        # SAM画像埋め込みがSAMデバイスにあることを確認
                                        sam_image_embedding_on_device = sam_image_embedding.to(sam_device)
                                        
                                        # SAMによるマスク生成
                                        print(f"SAM入力: image_embeddings={sam_image_embedding_on_device.shape}, sparse_embeddings={sparse_embeddings.shape}")
                                        
                                        mask_predictions, _ = self.sam.mask_decoder(
                                            image_embeddings=sam_image_embedding_on_device,
                                            image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings=sparse_embeddings,
                                            dense_prompt_embeddings=None,
                                            multimask_output=False,
                                        )
                                        
                                        # マスク予測を取得して後処理
                                        mask_pred = mask_predictions[0]  # [1, 1, H, W]
                                        
                                        # nan値やinf値がないか確認
                                        if torch.isnan(mask_pred).any() or torch.isinf(mask_pred).any():
                                            print("警告: マスク予測にNaNまたはInf値が含まれています")
                                            # NaNやInfを0に置き換え
                                            mask_pred = torch.nan_to_num(mask_pred, nan=0.0, posinf=1.0, neginf=0.0)
                                        
                                        mask_pred = torch.sigmoid(mask_pred)
                                        mask_binary = (mask_pred > 0.5).float()
                                        mask_np = mask_binary[0, 0].cpu().numpy()
                                        masks.append(mask_np)
                                        
                                        print(f"マスク {len(masks)} を生成しました: 形状={mask_np.shape}")
                                    
                                    except Exception as mask_error:
                                        print(f"SAMマスク生成中にエラー: {str(mask_error)}")
                                        import traceback
                                        traceback.print_exc()
                                        
                                        # デバッグ情報
                                        print(f"SAM画像埋め込み: 形状={sam_image_embedding.shape}, デバイス={sam_image_embedding.device}, dtype={sam_image_embedding.dtype}")
                                        print(f"スパース埋め込み: 形状={sparse_embeddings.shape}, デバイス={sparse_embeddings.device}, dtype={sparse_embeddings.dtype}")
                                        
                                        # 空のダミーマスクを追加（失敗した場合）
                                        if sam_image_embedding.shape[0] > 0:
                                            h, w = sam_image_embedding.shape[-2] * 4, sam_image_embedding.shape[-1] * 4
                                            dummy_mask = np.zeros((h, w), dtype=np.float32)
                                            masks.append(dummy_mask)
                                            print(f"ダミーマスクを追加しました: 形状={dummy_mask.shape}")
                                else:
                                    print("隠れ状態が見つかりません: outputs.hidden_statesがありません")
                        
                        except Exception as e:
                            print(f"マスク生成中にエラーが発生しました: {str(e)}")
                            import traceback
                            traceback.print_exc()
                
                except Exception as e:
                    print(f"トークン生成中にエラーが発生しました: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    # 最低限のテキスト応答を返す
                    return {"masks": masks, "text": f"エラーが発生しました: {str(e)}"}

            except Exception as e:
                print(f"画像とテキストの処理中にエラーが発生しました: {str(e)}")
                import traceback
                traceback.print_exc()
                return {"masks": [], "text": f"画像処理エラー: {str(e)}"}

        except Exception as e:
            print(f"セグメンテーション生成中にエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"masks": [], "text": f"全体的なエラー: {str(e)}"}
        
        # 特殊トークンを除外した最終テキストを生成
        if 'clean_text' in locals():
            final_text = clean_text
        elif 'decoded_text' in locals():
            # 特殊トークンを手動で除外
            final_text = decoded_text.replace(self.seg_token, "")
            # その他の特殊トークンも除外
            for token in ["<s>", "</s>", "<unk>"]:
                final_text = final_text.replace(token, "")
        else:
            final_text = "テキスト生成に失敗しました。"
        
        return {"masks": masks, "text": final_text.strip()}

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
