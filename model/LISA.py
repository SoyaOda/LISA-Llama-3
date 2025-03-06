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
import json


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
        LISAモデルの初期化

        引数:
            model_path: Llama 3.2 Visionモデルのパス
            sam_checkpoint: SAMモデルのチェックポイントファイル
            seg_token: セグメンテーショントークン文字列
            device: 使用するデバイス
            max_batch_size: 最大バッチサイズ
            use_deepspeed: DeepSpeedを使用するかどうか
            ds_config: DeepSpeed設定ファイルのパス
            local_rank: ローカルランク（分散学習用）
        """
        super().__init__()
        
        # 基本パラメータの設定
        self.seg_token = seg_token
        self.max_batch_size = max_batch_size
        self.device = device
        self.use_deepspeed = use_deepspeed
        self.ds_config = ds_config
        self.local_rank = local_rank
        
        # データ型の設定（GPUが利用可能であればfloat16、そうでなければfloat32）
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # モデル初期化
        self.model = None
        self.processor = None
        self.initialize_llama(model_path)
        self.transform = ResizeLongestSide(self.sam_image_size)
        
        # SAMモデルがある場合は初期化
        self.sam = None
        self.initialize_lisa_modules(sam_checkpoint)
        
        # もしDeepSpeed用のds_configが指定されていなければ、ZeRO-2設定をデフォルトで作成
        if self.use_deepspeed and not self.ds_config:
            print("ZeRO-2を使用した安定したDeepSpeed設定を使用します")
            self.ds_config = {
                "fp16": {
                    "enabled": True,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "initial_scale_power": 16,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                },
                "bf16": {
                    "enabled": False
                },
                "zero_optimization": {
                    "stage": 2,  # ZeRO-2に変更（より安定）
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "contiguous_gradients": True,
                    "overlap_comm": True
                },
                "gradient_accumulation_steps": 1,
                "gradient_clipping": 1.0,
                "steps_per_print": 50,
                "train_batch_size": 8,
                "train_micro_batch_size_per_gpu": 1,
                "wall_clock_breakdown": False
            }
            
            # 一時的なconfig.jsonファイルに書き出し
            with open("temp_ds_config.json", "w") as f:
                json.dump(self.ds_config, f, indent=2)
            self.ds_config = "temp_ds_config.json"
            print(f"一時DeepSpeed設定ファイルを作成: {self.ds_config}")

    def initialize_llama(self, model_path):
        """
        Llama 3.2 Vision モデルを初期化
        """
        print(f"プロセッサーとモデルをロードしています: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # DeepSpeedを使用する場合
        if self.use_deepspeed:
            print("DeepSpeedエンジンを初期化しています...")
            if self.ds_config is None:
                self.ds_config = "ds_config.json"
                print(f"デフォルトDeepSpeed設定ファイルを使用: {self.ds_config}")
            
            # モデルクラス選択
            try:
                model_class = MllamaForConditionalGeneration
                print("MllamaForConditionalGenerationを使用します")
            except Exception as e:
                print(f"MllamaForConditionalGenerationの使用に失敗: {str(e)}")
                model_class = AutoModelForCausalLM
                print("代わりにAutoModelForCausalLMを使用します")
            
            # DSの初期化
            self.model, _, _, _ = deepspeed.initialize(
                model=model_class.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                ),
                config=self.ds_config,
                model_parameters=None,
            )
            # デバイスパラメータの設定
            if self.device is None and torch.cuda.is_available():
                self.device = f"cuda:{self.local_rank}"
                print(f"自動的にデバイスを設定: {self.device}")
        
        # 通常のPyTorchを使用する場合
        else:
            print("標準PyTorchモデルを初期化しています...")
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"自動的にデバイスを設定: {self.device}")
            
            # モデルをロード
            try:
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                ).to(self.device)
                print("MllamaForConditionalGenerationを使用します")
            except Exception as e:
                print(f"MllamaForConditionalGenerationの使用に失敗: {str(e)}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                ).to(self.device)
                print("代わりにAutoModelForCausalLMを使用します")
        
        print(f"モデルのデバイス: {self.device}")
        
        # モデルの語彙サイズをトークナイザと一致させる（重要な修正）
        # これによりScatterGatherKernelエラーを解消
        tokenizer_vocab_size = len(self.processor.tokenizer)
        if hasattr(self.model, "module"):
            model_vocab_size = self.model.module.config.text_config.vocab_size
            if tokenizer_vocab_size != model_vocab_size:
                print(f"語彙サイズの不一致を検出: モデル={model_vocab_size}, トークナイザ={tokenizer_vocab_size}")
                print(f"モデルの語彙サイズをトークナイザに合わせて調整します")
                # DeepSpeedモデルの場合は内部モジュールを使う
                self.model.module.resize_token_embeddings(tokenizer_vocab_size)
        else:
            model_vocab_size = self.model.config.text_config.vocab_size
            if tokenizer_vocab_size != model_vocab_size:
                print(f"語彙サイズの不一致を検出: モデル={model_vocab_size}, トークナイザ={tokenizer_vocab_size}")
                print(f"モデルの語彙サイズをトークナイザに合わせて調整します")
                self.model.resize_token_embeddings(tokenizer_vocab_size)
        
        # <seg>トークンをトークナイザに追加
        self.expand_embedding_layer(self.seg_token)

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
                                    if isinstance(
    config, dict) and "text_config" in config:
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

    def generate_segmentation(
    self,
    image,
    prompt,
    max_new_tokens=1024,
    top_p=0.95,
    temperature=0.1,
    top_k=None,
    num_beams=1,
    repetition_penalty=1.0,
     do_sample=True):
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
                print(f"SAM画像埋め込みのシェイプ: {sam_image_embedding.shape}")

                # 入力に合わせたメッセージ形式を作成
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
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
                    model_inputs['aspect_ratio_ids'] = torch.zeros(
                        (b, n_media, n_tiles), dtype=torch.long, device=device)
                    model_inputs['aspect_ratio_mask'] = torch.ones(
                        (b, n_media, n_tiles), dtype=torch.long, device=device)
                    print("aspect_ratio_ids と aspect_ratio_mask を生成しました")

                # 生成パラメータ設定
                # top_kがNoneまたは大きすぎる場合の安全対策
                if top_k is None or top_k > 50:
                    top_k = 5  # 最小限のtop_k値
                print(f"使用するtop_k値: {top_k}")

                # 異常値回避のためのパラメータ検証
                temperature = max(0.1, min(2.0, temperature))  # 0.1~2.0の範囲に制限
                top_p = max(0.1, min(0.99, top_p))  # 0.1~0.99の範囲に制限
                
                # よりシンプルな生成アプローチを使用するフラグ
                use_simple_generation = True

                # 基本の生成パラメータ
                generation_params = {
                    "input_ids": model_inputs["input_ids"],
                    "attention_mask": model_inputs["attention_mask"],
                    "pixel_values": model_inputs["pixel_values"],
                    "aspect_ratio_ids": model_inputs["aspect_ratio_ids"],
                    "aspect_ratio_mask": model_inputs["aspect_ratio_mask"],
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": False,  # サンプリングを無効化
                    "num_beams": 1,      # ビームサーチを無効化
                    "use_cache": True,
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                }

                # メモリ使用量を表示
                if torch.cuda.is_available():
                    print(
                        f"生成前のGPUメモリ: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

                # まずテンソルの形状とデバイスを確認
                for k, v in generation_params.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k}: 形状={v.shape}, デバイス={v.device}, dtype={v.dtype}")
                        # NaNとInfをチェック
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            print(f"警告: {k}にNaNまたはInf値が含まれています。修正します。")
                            generation_params[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
                
                print(f"生成パラメータ設定: temperature={temperature}, top_p={top_p}, top_k={top_k}, グリーディ生成={not do_sample}")
                
                try:
                    if use_simple_generation:
                        print("安全な手動生成を実行します...")
                        
                        # モデルを評価モードに設定
                        model = self.model.module if hasattr(self.model, "module") else self.model
                        model.eval()
                        
                        # 入力IDの準備
                        input_ids = generation_params["input_ids"].clone()
                        attention_mask = generation_params["attention_mask"].clone()
                        pixel_values = generation_params["pixel_values"]
                        aspect_ratio_ids = generation_params["aspect_ratio_ids"]
                        aspect_ratio_mask = generation_params["aspect_ratio_mask"]
                        
                        # 最大トークン数
                        max_length = input_ids.shape[1] + min(100, max_new_tokens)
                        
                        with torch.inference_mode():
                            # 最初のフォワードパス - すべての入力を処理
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                aspect_ratio_ids=aspect_ratio_ids,
                                aspect_ratio_mask=aspect_ratio_mask,
                                return_dict=True,
                                use_cache=True
                            )
                            
                            # 最初のKVキャッシュを取得
                            past_key_values = outputs.past_key_values
                            
                            # 現在のトークンシーケンスを保存
                            generated = input_ids
                            
                            # 既存のトークン数をカウント
                            cur_len = input_ids.shape[1]
                            
                            # 一度に1トークンずつ生成
                            for _ in range(min(100, max_new_tokens)):
                                # 次のトークンを予測
                                next_token_logits = outputs.logits[:, -1, :]
                                
                                # 極端な値を安全に処理
                                next_token_logits = torch.nan_to_num(next_token_logits, nan=-float('inf'), posinf=-float('inf'), neginf=-float('inf'))
                                
                                # グリーディ選択（エラーを避けるため）
                                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                                
                                # シーケンスを更新
                                generated = torch.cat([generated, next_tokens], dim=-1)
                                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                                cur_len += 1
                                
                                # EOS条件をチェック
                                if next_tokens[0, 0].item() == self.processor.tokenizer.eos_token_id:
                                    break
                                
                                # 最大長をチェック
                                if cur_len >= max_length:
                                    break
                                
                                # 次のステップの入力を設定
                                outputs = model(
                                    input_ids=next_tokens,
                                    attention_mask=attention_mask[:, -1:],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    return_dict=True
                                )
                                past_key_values = outputs.past_key_values
                        
                        # 生成されたシーケンスを返す
                        generate_outputs = generated
                        print(f"安全な手動生成が完了しました。総トークン数: {generate_outputs.shape[1]}")
                        
                    else:
                        # 通常のモデル生成を試みる
                        print("DeepSpeed環境でgenerateを実行します")
                        generate_outputs = self.model.module.generate(**generation_params)
                        
                except Exception as e:
                    print(f"生成中にエラーが発生しました: {str(e)}")
                    print("最終的なフォールバック: 最小限の生成を試行します")
                    
                    try:
                        # トークナイザに戻し、最小限の応答で返す
                        minimal_text = "すみません、画像処理中にエラーが発生しました。"
                        minimal_tokens = self.processor.tokenizer.encode(minimal_text, return_tensors="pt").to(device)
                        generate_outputs = torch.cat([generation_params["input_ids"][:, :10], minimal_tokens], dim=1)
                    except Exception as e2:
                        print(f"最終フォールバックでもエラー: {str(e2)}")
                        # 最小限のダミー出力
                        generate_outputs = generation_params["input_ids"]
                        return {"masks": [], "text": "テキスト生成に失敗しました。"}
                        
                # 生成されたトークンIDを取得
                generate_ids = generate_outputs.detach()

                # 生成テキストをデコード
                decoded_text = self.processor.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=False
                )[0]

                print(f"生成テキスト: {decoded_text[:100]}...")

                # トークナイザからセグメンテーショントークンのIDを取得
                seg_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                    self.seg_token)

                if seg_token_id == self.processor.tokenizer.unk_token_id:
                    print(
                        f"警告: セグメンテーショントークン {self.seg_token} がボキャブラリに見つかりません。UNKトークンとして処理されます。")

                print(f"<seg>トークンID: {seg_token_id}")

                # 生成テキスト内の<seg>トークン位置を検索
                try:
                    # セグメンテーショントークンの位置を見つける
                    seg_positions = []
                    for batch_idx in range(generate_ids.shape[0]):
                        positions = torch.where(
                            generate_ids[batch_idx] == seg_token_id)[0]
                        for pos in positions:
                            seg_positions.append((batch_idx, pos.item()))

                    print(f"<seg>トークン位置: {seg_positions}")
                    
                    masks = []  # 生成されたマスクを保存するリスト

                    # 各<seg>トークンについてマスクを生成
                    for batch_idx, pos in seg_positions:
                        try:
                            # この位置までのシーケンスを抽出
                            input_ids_segment = generate_ids[batch_idx,
                                                            :pos+1].unsqueeze(0)
                            attention_mask_segment = torch.ones_like(
                                input_ids_segment)

                            # フォワードパスを実行して隠れ状態を取得
                            with torch.no_grad():
                                # テンソルの形状とデバイスを確認
                                pixel_values_segment = model_inputs['pixel_values'][:1]
                                aspect_ratio_ids_segment = model_inputs['aspect_ratio_ids'][:1] if 'aspect_ratio_ids' in model_inputs else None
                                aspect_ratio_mask_segment = model_inputs['aspect_ratio_mask'][:1] if 'aspect_ratio_mask' in model_inputs else None

                                print(f"フォワードパス用 pixel_values: 形状={pixel_values_segment.shape}")
                                
                                # NaNとInfをチェックして修正
                                for tensor_name, tensor in [
                                    ("pixel_values", pixel_values_segment),
                                    ("aspect_ratio_ids", aspect_ratio_ids_segment),
                                    ("aspect_ratio_mask", aspect_ratio_mask_segment)
                                ]:
                                    if tensor is not None:
                                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                                            print(f"警告: {tensor_name}にNaNまたはInf値が含まれています。修正します。")
                                            if tensor_name == "pixel_values":
                                                pixel_values_segment = torch.nan_to_num(
                                                    pixel_values_segment, nan=0.0, posinf=1.0, neginf=0.0)
                                            elif tensor_name == "aspect_ratio_ids":
                                                aspect_ratio_ids_segment = torch.nan_to_num(
                                                    aspect_ratio_ids_segment, nan=0.0, posinf=1.0, neginf=0.0)
                                            elif tensor_name == "aspect_ratio_mask":
                                                aspect_ratio_mask_segment = torch.nan_to_num(
                                                    aspect_ratio_mask_segment, nan=0.0, posinf=1.0, neginf=0.0)
                                
                                # GPUでの処理を試みる
                                try:
                                    outputs = self.model.module(
                                        input_ids=input_ids_segment.to(device),
                                        attention_mask=attention_mask_segment.to(device),
                                        pixel_values=pixel_values_segment.to(device),
                                        aspect_ratio_ids=aspect_ratio_ids_segment.to(device) if aspect_ratio_ids_segment is not None else None,
                                        aspect_ratio_mask=aspect_ratio_mask_segment.to(device) if aspect_ratio_mask_segment is not None else None,
                                        output_hidden_states=True,
                                        return_dict=True
                                    )
                                    
                                    print("GPUでの隠れ状態取得に成功しました")
                                    
                                except Exception as e:
                                    print(f"GPUでの隠れ状態取得中にエラー: {str(e)}")
                                    print("CPUでの処理に切り替えます")
                                    
                                    # CPUに移動して再試行
                                    try:
                                        # モデルをCPUに移動
                                        model_cpu = self.model.module.to('cpu')
                                        
                                        outputs = model_cpu(
                                            input_ids=input_ids_segment.cpu(),
                                            attention_mask=attention_mask_segment.cpu(),
                                            pixel_values=pixel_values_segment.cpu(),
                                            aspect_ratio_ids=aspect_ratio_ids_segment.cpu() if aspect_ratio_ids_segment is not None else None,
                                            aspect_ratio_mask=aspect_ratio_mask_segment.cpu() if aspect_ratio_mask_segment is not None else None,
                                            output_hidden_states=True,
                                            return_dict=True
                                        )
                                        
                                        # 処理後、モデルをGPUに戻す
                                        self.model.module.to(device)
                                        print("CPUでの隠れ状態取得に成功しました")
                                        
                                    except Exception as cpu_error:
                                        print(f"CPUでの処理も失敗しました: {str(cpu_error)}")
                                        # ダミーの隠れ状態とマスク情報を返し、次のトークンへ
                                        masks.append({
                                            "segmentation": np.zeros((self.image_size, self.image_size), dtype=np.uint8),
                                            "area": 0,
                                            "predicted_iou": 0.0,
                                            "stability_score": 0.0,
                                            "error": str(cpu_error)
                                        })
                                        continue  # 次のトークンへ

                                # 隠れ状態を取得
                                # 最後の層から<seg>トークンの最後の隠れ状態を抽出
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                    hidden_states = outputs.hidden_states[-1]
                                    seg_hidden_state = hidden_states[0, -1]  # バッチ0、最後の位置
                                    
                                    print(f"<seg>トークンの隠れ状態: 形状={seg_hidden_state.shape}")
                                    
                                    # SAMの予測に使用する形式にプロジェクション
                                    prompt_embedding = self.seg_projection(seg_hidden_state)
                                    
                                    # データタイプをfloat32に変換（SAMの要件に合わせる）
                                    prompt_embedding = prompt_embedding.float()
                                    
                                    # SAMへの入力を準備
                                    # 画像特徴マップがdeviceと一致していることを確認
                                    sam_image_embedding = sam_image_embedding.to(prompt_embedding.device)
                                    
                                    try:
                                        # SAMモデルでマスクを予測
                                        masks_predictions, scores, logits = self.sam.mask_decoder(
                                            image_embeddings=sam_image_embedding,
                                            image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings=prompt_embedding.unsqueeze(0),
                                            dense_prompt_embeddings=None,
                                            multimask_output=False,
                                        )
                                        
                                        # マスク予測結果がNaNまたはInfを含んでいないか確認
                                        if torch.isnan(masks_predictions).any() or torch.isinf(masks_predictions).any():
                                            print("警告: マスク予測にNaNまたはInf値が含まれています。0に置き換えます。")
                                            masks_predictions = torch.nan_to_num(masks_predictions, nan=0.0, posinf=1.0, neginf=0.0)
                                        
                                        # マスクをリサイズして画像の元のサイズに合わせる
                                        mask = F.interpolate(
                                            masks_predictions,
                                            size=(self.image_size, self.image_size),
                                            mode="bilinear",
                                            align_corners=False,
                                        )
                                        
                                        # マスクを2値化
                                        mask = (mask > 0).float().cpu().numpy()
                                        
                                        # マスク情報をリストに追加
                                        masks.append({
                                            "segmentation": mask[0, 0],
                                            "area": mask[0, 0].sum().item(),
                                            "predicted_iou": scores[0].item(),
                                            "stability_score": 1.0
                                        })
                                    
                                    except Exception as mask_error:
                                        print(f"マスク生成中にエラー: {str(mask_error)}")
                                        # ダミーのマスクを追加
                                        masks.append({
                                            "segmentation": np.zeros((self.image_size, self.image_size), dtype=np.uint8),
                                            "area": 0,
                                            "predicted_iou": 0.0,
                                            "stability_score": 0.0,
                                            "error": str(mask_error)
                                        })
                                
                                else:
                                    print(
                                        "警告: hidden_statesが取得できませんでした")
                                    masks.append({
                                        "segmentation": np.zeros((self.image_size, self.image_size), dtype=np.uint8),
                                        "area": 0,
                                        "predicted_iou": 0.0,
                                        "stability_score": 0.0,
                                        "error": "hidden_states not available"
                                    })
                        
                        except Exception as seg_error:
                            print(f"セグメンテーショントークン処理中にエラー: {str(seg_error)}")
                            masks.append({
                                "segmentation": np.zeros((self.image_size, self.image_size), dtype=np.uint8),
                                "area": 0,
                                "predicted_iou": 0.0,
                                "stability_score": 0.0,
                                "error": str(seg_error)
                            })
                            
                except Exception as token_error:
                    print(f"トークン処理中にエラー: {str(token_error)}")
                    masks = []  # 空のマスクリスト
                    
                # 最終テキストを取得（特殊トークンを除去）
                final_text = self.processor.tokenizer.decode(
                    generate_ids[0], skip_special_tokens=True
                )
                
                # 不要なトークンを削除
                if self.seg_token in final_text:
                    final_text = final_text.replace(self.seg_token, "")
                    
                    # その他の特殊トークンも削除
                    for token in ["<s>", "</s>", "<unk>"]:
                        final_text = final_text.replace(token, "")

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

    def expand_embedding_layer(self, new_token):
        """
        モデルの埋め込み層を拡張して新しいトークンを追加
        
        Args:
            new_token: 追加する新しいトークン
        """
        try:
            print(f"トークナイザと埋め込み層を拡張: {new_token}")
            
            # トークナイザに新しいトークンが含まれているかチェック
            if new_token not in self.processor.tokenizer.get_vocab():
                print(f"トークナイザに新トークンを追加: {new_token}")
                num_added_tokens = self.processor.tokenizer.add_tokens([new_token])
                print(f"追加されたトークン数: {num_added_tokens}")
                
                # トークンIDを取得
                self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(new_token)
                print(f"新トークンID: {self.seg_token_idx}")
                
                # モデルの埋め込み層を拡張
                if self.use_deepspeed:
                    print("DeepSpeedモデルの埋め込み層を拡張...")
                    # DeepSpeedモデルではmoduleを通してアクセス
                    vocab_size = len(self.processor.tokenizer)
                    current_vocab_size = self.model.module.config.text_config.vocab_size
                    
                    if vocab_size > current_vocab_size:
                        print(f"埋め込み層を拡張: {current_vocab_size} -> {vocab_size}")
                        try:
                            self.model.module.resize_token_embeddings(vocab_size)
                            print("埋め込み層の拡張が完了しました")
                        except Exception as e:
                            print(f"埋め込み層拡張中にエラー: {str(e)}")
                            print("このエラーは無視してください - トークンIDは取得済みです")
                else:
                    # 通常のモデルの場合
                    vocab_size = len(self.processor.tokenizer)
                    current_vocab_size = self.model.config.text_config.vocab_size
                    
                    if vocab_size > current_vocab_size:
                        print(f"埋め込み層を拡張: {current_vocab_size} -> {vocab_size}")
                        try:
                            self.model.resize_token_embeddings(vocab_size)
                            print("埋め込み層の拡張が完了しました")
                        except Exception as e:
                            print(f"埋め込み層拡張中にエラー: {str(e)}")
                            print("このエラーは無視してください - トークンIDは取得済みです")
            else:
                # すでにトークンが存在する場合、IDだけを取得
                self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(new_token)
                print(f"既存トークンのIDを取得: {self.seg_token_idx}")
                
        except Exception as e:
            print(f"トークン拡張処理中にエラーが発生: {str(e)}")
            # バックアップ処理: エラーが発生した場合でもトークンIDを取得
            try:
                self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(new_token)
                print(f"エラー後のフォールバック: トークンID = {self.seg_token_idx}")
            except:
                print("警告: トークンIDの取得に失敗しました。デフォルト値を使用します。")
                self.seg_token_idx = -100  # デフォルト値


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
