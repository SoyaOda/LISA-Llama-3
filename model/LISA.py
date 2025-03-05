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
                            # DeepSpeedエンジンのモデルに直接アクセス
                            ds_model = self.model.module
                            ds_model.resize_token_embeddings(vocab_size)
                        else:
                            # 通常の場合
                            self.model.resize_token_embeddings(vocab_size)
                            
                        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
                        print(f"セグメンテーショントークンのインデックス: {self.seg_token_idx}")
                        print("embedding層の拡張が完了しました")
                except Exception as e:
                    print(f"embeddings拡張中にエラー: {e}")
                    print("代替手段を試行...")
                    # 既存の重みを保持して拡張する代替手法
                    if self.use_deepspeed:
                        old_embeddings = self.model.module.get_input_embeddings()
                    else:
                        old_embeddings = self.model.get_input_embeddings()
                        
                    old_num_tokens = old_embeddings.weight.size(0)
                    new_num_tokens = len(self.processor.tokenizer)

                    if new_num_tokens > old_num_tokens:
                        # 新しいembedding行列を作成し、古い重みをコピー
                        new_embeddings = nn.Embedding(
                            new_num_tokens, old_embeddings.weight.size(1))
                        new_embeddings.to(
                            old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

                        # 既存の重みを保持
                        with torch.no_grad():
                            new_embeddings.weight[:old_num_tokens,
                                                  :] = old_embeddings.weight[:, :]

                            # 新しいトークンの重みを最後の10トークンの平均で初期化
                            avg_weight = old_embeddings.weight[-10:,
                                                               :].mean(dim=0, keepdim=True)
                            new_embeddings.weight[old_num_tokens:,
                                                  :] = avg_weight

                        # 新しいembeddingに置き換え
                        if self.use_deepspeed:
                            self.model.module.set_input_embeddings(new_embeddings)
                        else:
                            self.model.set_input_embeddings(new_embeddings)
                            
                        print("代替手法によるembedding拡張が完了しました")

            # セグメンテーショントークンのインデックスを取得
            self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(
                self.seg_token)
            print(f"セグメンテーショントークンID: {self.seg_token_idx}")

        except Exception as e:
            print(f"モデルのロード中にエラーが発生: {e}")
            raise

    def initialize_lisa_modules(self, sam_checkpoint=None):
        """
        LISAの追加モジュール（SAM、セグメンテーション投影など）を初期化
        """
        # SAMチェックポイントがあれば、SAMをロード
        if sam_checkpoint:
            print(f"SAMチェックポイント: {sam_checkpoint}")
            try:
                print(f"SAMチェックポイントを読み込みます: {sam_checkpoint}")
                self.sam = build_sam_vit_h(checkpoint=sam_checkpoint)

                # SAMモデルをGPUに移動
                print(f"SAMモデルを {self.device} デバイスに移動します...")
                self.sam.to(self.device)

                # デバイスがCUDAの場合、SAMモデルを半精度に変換
                if self.device == "cuda" and self.dtype == torch.float16:
                    print("SAMモデルを半精度（float16）に変換します...")
                    # モデル全体を半精度に変換
                    self.sam = self.sam.half()

                    # すべてのパラメータが確実に半精度になるよう明示的に変換
                    for name, param in self.sam.named_parameters():
                        param.data = param.data.half()
                        if param._grad is not None:
                            param._grad.data = param._grad.data.half()

                    # すべてのバッファも半精度に変換
                    for name, buf in self.sam.named_buffers():
                        buf.data = buf.data.half()

                    print("SAMモデルを半精度に変換しました")

                # SAMパラメータを凍結
                print("SAMのパラメータを凍結します...")
                for param in self.sam.parameters():
                    param.requires_grad = False

                print("SAMモデルが正常に初期化されました")

                # テキスト->SAMプロンプト投影（Llama 3.2 Visionの隠れ状態次元から256次元へ）
                # Llama 3.2 Visionのhidden_sizeはtext_configのhidden_sizeから取得
                hidden_size = self.model.config.text_config.hidden_size

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
        SAMモデル用に画像を前処理します。

        Args:
            image: PIL画像

        Returns:
            image_embedding: SAMのイメージエンコーダからの埋め込み
            original_image_size: 元の画像サイズ
        """
        try:
            print("SAM前処理を実行中...")
            h, w = image.shape[:2]
            print(f"SAM変換を初期化しました (target_size=1024)")
            transform = ResizeLongestSide(long_side_length=1024)
            input_image = transform.apply_image(image)

            # 画像パディング情報を出力
            input_size = input_image.shape[:2]
            print(
                f"画像をパディングしました: 1024x{int(1024*h/w)} -> {input_size[0]}x{input_size[1]}")

            input_image_torch = torch.as_tensor(
                input_image, device=self.device)
            input_image_torch = input_image_torch.permute(
                2, 0, 1).contiguous()[None, :, :, :]
            print(f"SAM入力画像のシェイプ: {input_image_torch.shape}")

            # デバイスの明示的な指定
            if hasattr(self.sam, 'image_encoder'):
                self.sam.image_encoder.to(self.device)

                # 位置埋め込みの補間を詳細に出力
                orig_embed_size = self.sam.image_encoder.pos_embed.shape[1:3]
                target_embed_size = (
                    input_image_torch.shape[2] // 16, input_image_torch.shape[3] // 16)
                print(
                    f"位置埋め込みを補間します: {orig_embed_size} -> {target_embed_size}")

                with torch.no_grad():
                    image_embedding = self.sam.image_encoder(input_image_torch)
                    print(f"SAM画像埋め込みのシェイプ: {image_embedding.shape}")

                return image_embedding, image.shape[:2]
            else:
                raise ValueError("SAMモデルのimage_encoderが初期化されていません")
        except Exception as e:
            print(f"SAM前処理を実行中にエラーが発生: {str(e)}")
            raise e

    def generate_segmentation(self, image, text_prompt, **kwargs):
        """
        画像とプロンプトからセグメンテーションを生成する
        image: PIL.Image - 入力画像
        prompt: str - 入力プロンプト
        max_new_tokens: int - 生成するトークンの最大数
        generation_params: dict - generate関数に渡す追加パラメータ
        """
        try:
            # DeepSpeed使用時はデバイス設定をスキップ（既にDeepSpeedが管理）
            if not self.use_deepspeed:
                # ここで明示的にデバイスをCPUに設定
                device = torch.device("cpu")
                self.device = device
                print(f"デバイス: {self.device}")

                # デバイスの一貫性を確保
                if hasattr(self, 'model'):
                    self.model.to(device)
                if hasattr(self, 'sam'):
                    self.sam.to(device)
                if hasattr(self, 'seg_projection'):
                    self.seg_projection.to(device)
            else:
                print("DeepSpeed使用中: デバイス設定はスキップされます")

            # 生成パラメータ（デフォルト値を設定）
            temperature = kwargs.get('temperature', 0.2)
            do_sample = kwargs.get('do_sample', True)
            top_p = kwargs.get('top_p', 0.7)
            repetition_penalty = kwargs.get('repetition_penalty', 1.2)
            num_beams = kwargs.get('num_beams', 1)
            max_new_tokens = kwargs.get('max_new_tokens', 256)

            print(f"生成パラメータ: {kwargs}")

            # 画像をSAMに通す前処理
            print("SAM前処理を実行中...")
            # SAM用の画像埋め込みを計算
            image_embedding, original_image_size = self.preprocess_sam_image(
                image)
            print(f"SAM画像埋め込みのシェイプ: {image_embedding.shape}")

            # メッセージの作成
            print("チャットメッセージを作成...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image"}
                    ]
                }
            ]
            print(f"生成するメッセージ: {messages}")

            # チャットテンプレートを適用
            print("チャットテンプレートを適用...")
            input_text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            print(f"フォーマット後の入力テキスト: {input_text[:100]}...")

            # プロセッサを使って入力を準備
            print("プロセッサで入力を準備中...")
            inputs = self.processor(
                image,
                input_text,
                return_tensors="pt"
            )

            # 入力テンソルの形状を表示
            for key, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"入力テンソル {key} の形状: {tensor.shape}")

            # DeepSpeed使用時はデバイス移動をスキップ（既にDeepSpeedが管理）
            if not self.use_deepspeed:
                # デバイスに送る
                inputs = {k: v.to(self.device) if isinstance(
                    v, torch.Tensor) else v for k, v in inputs.items()}
            
            # モデルのフォワードパスを実行して過去の状態を取得
            print("モデルのフォワードパスを実行して過去の状態を取得...")
            print(f"モデル入力のキー: {inputs.keys()}")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(
                        f"モデル入力 {key} の形状: {value.shape}, デバイス: {value.device}")

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
                    # generate メソッドの代わりに手動でデコードを行う
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    # 簡易的に次のトークンを生成
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)

                    # シンプルな実装：手動でトークンを生成
                    generated_ids = inputs["input_ids"]
                    max_new_tokens = kwargs.get("max_new_tokens", 30)

                    for _ in range(max_new_tokens):
                        current_inputs = {
                            "input_ids": generated_ids,
                            "attention_mask": torch.ones_like(generated_ids),
                        }

                        # 画像情報は最初の入力と同じものを使用
                        for k in inputs:
                            if k not in current_inputs and k != "input_ids" and k != "attention_mask":
                                current_inputs[k] = inputs[k]

                        outputs = self.model(**current_inputs)
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1)

                        # 終了トークンをチェック
                        if next_token.item() in self.processor.tokenizer.eos_token_id:
                            break

                        generated_ids = torch.cat(
                            [generated_ids, next_token.unsqueeze(0)], dim=1)

                        # セグメンテーショントークンをチェック
                        if next_token.item() == self.seg_token_idx:
                            # セグメンテーション処理
                            # 実装は略...
                            pass

                    decoded_text = self.processor.tokenizer.decode(
                        generated_ids[0], skip_special_tokens=False)
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
