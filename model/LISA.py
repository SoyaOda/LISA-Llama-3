#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
import os
import time
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
        max_batch_size=1
    ):
        """
        LISAモデルを初期化
        
        Args:
            model_path (str): Llama 3.2 Vision Instructモデルのパス
            sam_checkpoint (str): SAMチェックポイントのパス
            seg_token (str): セグメンテーショントークン
            device (str): 使用するデバイス ('cuda' または 'cpu')
            max_batch_size (int): 最大バッチサイズ
        """
        super().__init__()
        
        # デバイスを設定
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # セグメンテーショントークンを設定
        self.seg_token = seg_token
        
        # 各コンポーネントの初期化
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.sam = None
        self.seg_token_idx = None
        self.sam_transform = None
        
        # 半精度を使用
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"モデルの精度: {self.dtype}")
        
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
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                device_map=self.device,
                torch_dtype=self.dtype,
                trust_remote_code=True
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
                    # mean_resizing=Falseでkaimingの初期化を回避
                    self.model.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=False)
                    print("embedding層の拡張が完了しました")
                except Exception as e:
                    print(f"embeddings拡張中にエラー: {e}")
                    print("代替手段を試行...")
                    # 既存の重みを保持して拡張する代替手法
                    old_embeddings = self.model.get_input_embeddings()
                    old_num_tokens = old_embeddings.weight.size(0)
                    new_num_tokens = len(self.processor.tokenizer)
                    
                    if new_num_tokens > old_num_tokens:
                        # 新しいembedding行列を作成し、古い重みをコピー
                        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.weight.size(1))
                        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
                        
                        # 既存の重みを保持
                        with torch.no_grad():
                            new_embeddings.weight[:old_num_tokens, :] = old_embeddings.weight[:, :]
                            
                            # 新しいトークンの重みを最後の10トークンの平均で初期化
                            avg_weight = old_embeddings.weight[-10:, :].mean(dim=0, keepdim=True)
                            new_embeddings.weight[old_num_tokens:, :] = avg_weight
                        
                        # 新しいembeddingに置き換え
                        self.model.set_input_embeddings(new_embeddings)
                        print("代替手法によるembedding拡張が完了しました")
                
            # セグメンテーショントークンのインデックスを取得
            self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
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
                nn.init.kaiming_normal_(self.seg_projection.weight, nonlinearity='relu')
                
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
        print("SAM前処理を実行中...")
        
        # 画像をRGBに変換
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 元の画像サイズを保存
        original_image_size = image.size
        
        # 画像をSAM用に変換
        transform = self.transform_image()
        print(f"SAM変換を初期化しました (target_size={transform.target_length})")
        
        # 画像をテンソルに変換
        input_image = np.array(image)
        input_image_torch = transform.apply_image(input_image)
        input_image_torch = torch.as_tensor(input_image_torch, device=self.device)
        
        # 画像の形状を調整
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        # 必要に応じて半精度に変換
        if self.dtype == torch.float16 and self.device == "cuda":
            input_image_torch = input_image_torch.half()
        else:
            input_image_torch = input_image_torch.float()
            
        # 画像のパディング情報を取得
        input_size = tuple(input_image_torch.shape[-2:])
        padded_size = transform.get_preprocess_shape(*original_image_size[::-1], long_side_length=transform.target_length)
        print(f"画像をパディングしました: {padded_size[1]}x{padded_size[0]} -> {input_size[0]}x{input_size[1]}")
        
        print(f"SAM入力画像のシェイプ: {input_image_torch.shape}")
        
        # 画像エンコーダで埋め込みを計算
        try:
            image_embedding = self.sam.image_encoder(input_image_torch)
            return image_embedding, original_image_size
        except Exception as e:
            print(f"SAM画像エンコーダでエラーが発生: {str(e)}")
            raise e
    
    def generate_segmentation(self, image, prompt, **params):
        """
        画像とプロンプトからセグメンテーションを生成する
        image: PIL.Image - 入力画像
        prompt: str - 入力プロンプト
        max_new_tokens: int - 生成するトークンの最大数
        generation_params: dict - generate関数に渡す追加パラメータ
        """
        print(f"デバイス: {self.device}")
        
        try:
            # 生成パラメータ（デフォルト値を設定）
            temperature = params.get('temperature', 0.2)
            do_sample = params.get('do_sample', True)
            top_p = params.get('top_p', 0.7)
            repetition_penalty = params.get('repetition_penalty', 1.2)
            num_beams = params.get('num_beams', 1)
            max_new_tokens = params.get('max_new_tokens', 256)

            print(f"生成パラメータ: {params}")
            
            # 画像をSAMに通す前処理
            print("SAM前処理を実行中...")
            # SAM用の画像埋め込みを計算
            image_embedding, original_image_size = self.preprocess_sam_image(image)
            print(f"SAM画像埋め込みのシェイプ: {image_embedding.shape}")
            
            # メッセージの作成
            print("チャットメッセージを作成...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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
            
            # デバイスに送る
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # モデルのフォワードパスを実行して過去の状態を取得
            print("モデルのフォワードパスを実行して過去の状態を取得...")
            
            # すべての入力キーを表示
            print(f"モデル入力のキー: {inputs.keys()}")
            
            # 各入力の形状とデバイスを表示
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"モデル入力 {key} の形状: {value.shape}, デバイス: {value.device}")
                else:
                    print(f"モデル入力 {key} は {type(value)} 型です")
            
            # メモリを節約するためにキャッシュをクリア
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            try:
                # 小さなバッチサイズでフォワードパスを実行
                inputs["use_cache"] = True
                inputs["return_dict"] = True
                inputs["output_hidden_states"] = True
                
                # バッチサイズを1に固定
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 0:
                        inputs[key] = inputs[key][:1]
                
                # フォワードパスを実行
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.model(**inputs)
                
                past_key_values = outputs.past_key_values
                last_hidden_states = outputs.hidden_states[-1]
                
                # 最初のトークンの生成
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                # 生成パラメータを設定
                gen_params = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "num_beams": 1,  # 常に1に固定（メモリ節約）
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                }
                
                # メモリを節約するために不要な変数を削除
                del outputs
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                print("トークン生成を開始...")
                # トークン生成を小さなステップに分割して実行
                generated_ids = []
                current_input_ids = input_ids
                current_attention_mask = attention_mask
                current_past_key_values = past_key_values
                
                # 最大でmax_new_tokensまで生成
                with torch.no_grad():
                    for _ in range(min(max_new_tokens, 100)):  # 最大100トークンに制限
                        try:
                            # 1回の生成で1トークンのみ生成
                            current_gen_params = {
                                "input_ids": current_input_ids,
                                "attention_mask": current_attention_mask,
                                "past_key_values": current_past_key_values,
                                "do_sample": do_sample,
                                "temperature": temperature,
                                "top_p": top_p,
                                "repetition_penalty": repetition_penalty,
                                "max_new_tokens": 1,  # 1トークンずつ生成
                                "num_beams": 1,
                                "use_cache": True,
                                "pad_token_id": self.processor.tokenizer.pad_token_id,
                                "eos_token_id": self.processor.tokenizer.eos_token_id,
                                "return_dict_in_generate": True,
                                "output_scores": True,
                            }
                            
                            # 生成処理実行
                            outputs = self.model.generate(**current_gen_params)
                            
                            # 生成されたトークンを取得
                            new_token = outputs.sequences[0, -1].unsqueeze(0).unsqueeze(0)  # [1, 1]
                            
                            # SEGトークンをチェック
                            if new_token.item() == self.seg_token_idx:
                                print(f"SEGトークンが生成されました: {new_token.item()}")
                                # SEGトークンの隠れ状態を取得してSAMに渡す
                                seg_hidden_state = last_hidden_states[0, -1].unsqueeze(0)  # [1, hidden_size]
                                
                                # プロジェクションレイヤーを適用
                                seg_embedding = self.seg_projection(seg_hidden_state)  # [1, 256]
                                
                                # マスク予測
                                print("SAMマスク予測を実行中...")
                                masks, _, _ = self.sam.mask_decoder(
                                    image_embeddings=image_embedding,
                                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=seg_embedding,
                                    dense_prompt_embeddings=None,
                                    multimask_output=False,
                                )
                                
                                # マスクをリサイズして返す
                                mask = masks[0, 0].cpu().numpy()
                                mask = cv2.resize(mask, (original_image_size[1], original_image_size[0]))
                                mask = mask > 0.0
                                
                                return {
                                    "masks": [mask],
                                    "text": self.processor.tokenizer.decode(generated_ids + [new_token.item()], skip_special_tokens=False)
                                }
                            
                            # 生成されたトークンを追加
                            generated_ids.append(new_token.item())
                            
                            # EOSトークンが生成されたら終了
                            if new_token.item() == self.processor.tokenizer.eos_token_id:
                                print("EOSトークンが生成されました。生成を終了します。")
                                break
                            
                            # 入力を更新
                            current_input_ids = new_token
                            current_attention_mask = torch.ones_like(current_input_ids)
                            current_past_key_values = outputs.past_key_values
                            
                            # 20トークンごとにメモリをクリア
                            if len(generated_ids) % 20 == 0 and hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                                print(f"{len(generated_ids)}トークン生成済み...")
                                
                        except Exception as e:
                            print(f"トークン生成中にエラーが発生しました: {e}")
                            import traceback
                            traceback.print_exc()
                            break
                
                # 生成されたテキストを返す（SEGトークンなし）
                return {
                    "masks": [],
                    "text": self.processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
                }
                
            except Exception as e:
                print(f"フォワードパス実行中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                return {"masks": [], "text": f"エラー: {str(e)}"}
        
        except Exception as e:
            print(f"segmentation生成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return {"masks": [], "text": f"エラー: {str(e)}"}
    
    @classmethod
    def from_pretrained(cls, model_path, sam_checkpoint=None, **kwargs):
        """
        事前学習済みモデルからLISAモデルを作成
        """
        return cls(
            model_path=model_path,
            sam_checkpoint=sam_checkpoint,
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

