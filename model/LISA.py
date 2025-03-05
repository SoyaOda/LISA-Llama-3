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
                        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
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
                            self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
                            print(f"セグメンテーショントークンのインデックス: {self.seg_token_idx}")
                            print("トークンは追加されましたが、埋め込み拡張はスキップされました")
                            
                            # 注意：実際の運用では、埋め込み拡張を含むモデルを事前に保存し、
                            # それをDeepSpeedでロードすることを推奨します
                        else:
                            # 通常の場合
                            self.model.resize_token_embeddings(vocab_size)
                            
                            self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
                            print(f"セグメンテーショントークンのインデックス: {self.seg_token_idx}")
                            print("embedding層の拡張が完了しました")
                except Exception as e:
                    print(f"embeddings拡張中にエラー: {e}")
                    print("別の代替手段を試行...")
                    try:
                        # 最低限の対応として、トークンIDの取得のみ行う
                        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
                        print(f"トークン拡張はスキップしますが、セグメンテーショントークンIDを保存: {self.seg_token_idx}")
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
                                    print(f"text_configからhidden_size取得: {hidden_size}")
                                else:
                                    # dictの場合やtext_configが存在しない場合の対応
                                    if isinstance(config, dict) and "text_config" in config:
                                        hidden_size = config["text_config"]["hidden_size"]
                                        print(f"text_config dictからhidden_size取得: {hidden_size}")
                                    else:
                                        # フォールバック: 一般的なサイズを使用
                                        print("警告: モデル設定からhidden_sizeを取得できません。デフォルト値4096を使用します。")
                                        hidden_size = 4096  # Llama-3.2の一般的なサイズ
                            else:
                                print("警告: モデルmoduleにconfigがありません。デフォルト値4096を使用します。")
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
                original_size = (original_size[0], original_size[1])  # 明示的にタプルに変換
                
                print(f"オリジナル画像サイズ: {original_size}")
                
                # PIL画像をRGBに変換してからnumpy配列に変換
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # numpy配列に変換し、[H, W, C]形式にする
                input_image = np.array(image)
                print(f"入力画像のシェイプ: {input_image.shape}, 型: {input_image.dtype}")
                
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

    def generate_segmentation(self, image, prompt, **kwargs):
        """
        セグメンテーションを生成する関数
        
        Args:
            image: 入力画像（PIL.Image）
            prompt: セグメンテーションのプロンプト
            kwargs: 生成パラメータ（temperature, do_sample, top_p, repetition_penalty, num_beams, max_new_tokens）
            
        Returns:
            生成されたセグメンテーション結果
        """
        try:
            # デバイスの設定（DeepSpeed使用時は不要）
            device = next(self.model.parameters()).device if not self.use_deepspeed else None
            
            # 生成パラメータ
            temperature = kwargs.get("temperature", 0.1)
            do_sample = kwargs.get("do_sample", True)
            top_p = kwargs.get("top_p", 0.7)
            repetition_penalty = kwargs.get("repetition_penalty", 1.0)
            num_beams = kwargs.get("num_beams", 1)
            max_new_tokens = kwargs.get("max_new_tokens", 100)
            
            print(f"生成パラメータ: temperature={temperature}, do_sample={do_sample}, top_p={top_p}, "
                  f"repetition_penalty={repetition_penalty}, num_beams={num_beams}, max_new_tokens={max_new_tokens}")

            # SAM処理用に画像を前処理
            image_embedding = self.preprocess_sam_image(image)
            
            # チャットメッセージを作成
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # マニュアルでチャットテンプレートを構築
            # Llama 3.2 Visionのフォーマットに合わせる: <|begin_of_chat|><|user|><|image|>プロンプト<|assistant|>
            input_text = "<|begin_of_chat|>\n<|user|>\n<|image|>\n" + prompt + "\n<|assistant|>"
            print(f"構築されたテキスト入力: {input_text}")
            
            # 画像をPIL形式から変換
            if isinstance(image, Image.Image):
                # PIL画像をnumpy配列に変換
                image_np = np.array(image)
                # [H, W, C] -> [C, H, W]に変換
                image_np = np.transpose(image_np, (2, 0, 1))
                # バッチ次元を追加
                image_np = np.expand_dims(image_np, axis=0)
                # 画素値を0-255から0-1に正規化
                image_np = image_np / 255.0
                # numpyからTensorへ変換
                image_tensor = torch.from_numpy(image_np).float()
            else:
                # 既にテンソルの場合は正規化のみ
                if isinstance(image, torch.Tensor):
                    image_tensor = image.float() / 255.0 if image.max() > 1.0 else image.float()
                else:
                    raise ValueError("imageはPIL.ImageまたはTorch.Tensorである必要があります")

            # プロンプトをトークン化
            tokenized_text = self.processor.tokenizer(input_text, return_tensors="pt")
            
            # DeepSpeed使用時はテンソルをCPUからGPUに移動しない（DeepSpeedが管理）
            if not self.use_deepspeed:
                image_tensor = image_tensor.to(device)
                tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
            
            # 入力辞書を作成
            inputs = {
                "input_ids": tokenized_text["input_ids"],
                "attention_mask": tokenized_text["attention_mask"],
                "pixel_values": image_tensor,
                # アスペクト比IDとマスクはLlama 3.2 Vision仕様に合わせて計算（ここではダミー値）
                "aspect_ratio_ids": torch.zeros_like(tokenized_text["input_ids"]),
                "aspect_ratio_mask": torch.ones_like(tokenized_text["input_ids"]),
            }
            
            # 入力テンソルの形状を表示
            for key, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"入力テンソル {key} の形状: {tensor.shape}")
            
            try:
                # DeepSpeed使用時の処理
                if self.use_deepspeed:
                    print("DeepSpeedを使用してテキスト生成を実行...")
                    # DeepSpeedモデルでのgenerate呼び出し
                    generate_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"],
                        aspect_ratio_ids=inputs["aspect_ratio_ids"],
                        aspect_ratio_mask=inputs["aspect_ratio_mask"],
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                    
                    # トークンをデコード
                    decoded_text = self.processor.tokenizer.batch_decode(
                        generate_ids, skip_special_tokens=False
                    )[0]
                    
                    return decoded_text
                else:
                    # 非DeepSpeed環境での処理（既存のコード）
                    # generate メソッドを使用
                    generate_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"],
                        aspect_ratio_ids=inputs["aspect_ratio_ids"],
                        aspect_ratio_mask=inputs["aspect_ratio_mask"],
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                    
                    # トークンをデコード
                    decoded_text = self.processor.tokenizer.batch_decode(
                        generate_ids, skip_special_tokens=False
                    )[0]
                    
                    return decoded_text

            except Exception as e:
                print(f"トークン生成中にエラーが発生しました: {str(e)}")
                import traceback
                traceback.print_exc()
                return ""

        except Exception as e:
            print(f"segmentation生成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return {"masks": [], "text": f"エラー: {str(e)}"}

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
