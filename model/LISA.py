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
import shutil


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
        local_rank=-1,
        image_size=1024  # image_sizeをコンストラクタの引数に追加
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
            image_size: SAMの画像サイズと出力セグメンテーション画像サイズ (デフォルト: 1024)
        """
        super().__init__()
        
        # 基本パラメータの設定
        self.seg_token = seg_token
        self.max_batch_size = max_batch_size
        self.device = device
        self.use_deepspeed = use_deepspeed
        self.ds_config = ds_config
        self.local_rank = local_rank
        self.sam_image_size = image_size  # sam_image_sizeを初期化
        self.transform = None
        
        # データ型の設定（GPUが利用可能であればfloat16、そうでなければfloat32）
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.dtype = self.torch_dtype  # 別名
        
        # モデル初期化
        self.model = None
        self.processor = None
        self.initialize_llama(model_path)
        
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
        
        # <seg>トークンをトークナイザに追加 - 先にトークナイザに追加（埋め込み拡張前）
        print(f"トークナイザにセグメンテーショントークンを追加します: {self.seg_token}")
        num_added_tokens = self.processor.tokenizer.add_tokens([self.seg_token])
        print(f"追加されたトークン数: {num_added_tokens}")
        
        # セグメンテーショントークンのIDを取得
        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
        print(f"セグメンテーショントークンID: {self.seg_token_idx}")
        print(f"トークナイザの語彙サイズ: {len(self.processor.tokenizer)}")
        
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
            
            # モデルの初期化（DeepSpeed初期化前）
            print("モデルを一時的に初期化してトークナイザと同期します")
            temp_model = model_class.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            # リサイズ前にモデルの出力層と入力埋め込み層のサイズをチェック
            print(f"リサイズ前: トークナイザサイズ={len(self.processor.tokenizer)}")
            
            # モデル構造に基づいて埋め込み層をリサイズ
            try:
                # 新しい語彙サイズを取得
                new_vocab_size = len(self.processor.tokenizer)
                print(f"リサイズ前: トークナイザサイズ={new_vocab_size}")
                
                # 入力埋め込み層のリサイズ
                temp_model.resize_token_embeddings(new_vocab_size)
                print(f"入力埋め込み層をリサイズしました: {new_vocab_size}")
                
                # 出力埋め込み層の手動リサイズ
                if hasattr(temp_model, "get_output_embeddings") and temp_model.get_output_embeddings() is not None:
                    old_embeddings = temp_model.get_output_embeddings()
                    print(f"出力埋め込み層を取得: {old_embeddings}")
                    
                    # 拡張された出力埋め込み層を作成
                    old_num_tokens = old_embeddings.weight.shape[0]
                    if old_num_tokens < new_vocab_size:
                        print(f"出力埋め込み層をリサイズします: {old_num_tokens} -> {new_vocab_size}")
                        
                        # 新しい重みを作成
                        old_weights = old_embeddings.weight.data
                        new_weights = torch.zeros(
                            (new_vocab_size, old_weights.shape[1]),
                            dtype=old_weights.dtype,
                            device=old_weights.device
                        )
                        new_weights[:old_num_tokens, :] = old_weights
                        
                        # 新しいトークン埋め込みを平均値で初期化
                        if new_vocab_size > old_num_tokens:
                            num_new_tokens = new_vocab_size - old_num_tokens
                            mean_weights = old_weights.mean(dim=0).unsqueeze(0).expand(num_new_tokens, -1)
                            new_weights[old_num_tokens:, :] = mean_weights
                        
                        # 出力埋め込み層の重みを更新
                        old_embeddings.weight.data = new_weights
                        print(f"出力埋め込み層のリサイズが完了しました: {new_weights.shape}")
                else:
                    print("警告: 出力埋め込み層が見つからないか、アクセスできません")
                    
                    # text_modelにアクセスできるか確認
                    if hasattr(temp_model, "text_model"):
                        print("text_model経由で埋め込み層にアクセスを試みます")
                        if hasattr(temp_model.text_model, "lm_head") and temp_model.text_model.lm_head is not None:
                            lm_head = temp_model.text_model.lm_head
                            old_num_tokens = lm_head.weight.shape[0]
                            if old_num_tokens < new_vocab_size:
                                print(f"text_model.lm_headをリサイズします: {old_num_tokens} -> {new_vocab_size}")
                                # 同様のリサイズ処理...
                                old_weights = lm_head.weight.data
                                new_weights = torch.zeros(
                                    (new_vocab_size, old_weights.shape[1]),
                                    dtype=old_weights.dtype,
                                    device=old_weights.device
                                )
                                new_weights[:old_num_tokens, :] = old_weights
                                
                                # 新しいトークン埋め込みを平均値で初期化
                                if new_vocab_size > old_num_tokens:
                                    num_new_tokens = new_vocab_size - old_num_tokens
                                    mean_weights = old_weights.mean(dim=0).unsqueeze(0).expand(num_new_tokens, -1)
                                    new_weights[old_num_tokens:, :] = mean_weights
                                
                                # 出力埋め込み層の重みを更新
                                lm_head.weight.data = new_weights
                                print(f"text_model.lm_headのリサイズが完了しました: {new_weights.shape}")
                
                print(f"トークナイザサイズに合わせてモデルをリサイズしました: {new_vocab_size}")
            
            except Exception as resize_error:
                print(f"埋め込み層のリサイズ中にエラーが発生しました: {str(resize_error)}")
                print("警告: モデルは標準のトークナイザサイズのままです")
                
            # モデルをデバイスに移動（必要に応じて）
            print("モデルを適切なデバイスに移動します")
            if self.device is not None and not self.use_deepspeed:
                temp_model = temp_model.to(self.device)
                print(f"モデルを{self.device}に移動しました")
            
            # 一時モデルの状態をチェックポイントとして保存
            print("リサイズしたモデルを一時チェックポイントとして保存します")
            temp_save_dir = "./temp_resized_model"
            os.makedirs(temp_save_dir, exist_ok=True)
            temp_model.save_pretrained(temp_save_dir)
            del temp_model
            torch.cuda.empty_cache()
            
            # 保存したリサイズ済みモデルを使ってDeepSpeedを初期化
            print(f"リサイズしたモデルを使用してDeepSpeedを初期化: {temp_save_dir}")
            try:
                self.model, _, _, _ = deepspeed.initialize(
                    model=model_class.from_pretrained(
                        temp_save_dir,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        ignore_mismatched_sizes=True,  # サイズ不一致を無視
                    ),
                    config=self.ds_config,
                    model_parameters=None,
                )
                print("DeepSpeedの初期化が成功しました")
            except Exception as e:
                print(f"DeepSpeed初期化エラー: {str(e)}")
                print("標準モデルにフォールバックします")
                
                # エラー時のフォールバック: 標準モデルを使用
                self.model = model_class.from_pretrained(
                    temp_save_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    ignore_mismatched_sizes=True,  # サイズ不一致を無視
                ).to(self.device)
                print("標準モデルを使用します")
            
            # 一時チェックポイントを削除
            try:
                shutil.rmtree(temp_save_dir, ignore_errors=True)
                print("一時モデルチェックポイントを削除しました")
            except Exception as e:
                print(f"一時ディレクトリの削除エラー: {str(e)}")
            
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
                # まず標準のモデルをロード
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                    ignore_mismatched_sizes=True,  # サイズ不一致を無視
                )
                print("MllamaForConditionalGenerationを使用します")
                
                # デバッグ: モデル構造の詳細を出力
                print("\n===== MllamaForConditionalGeneration モデル構造 =====")
                print(f"モデルクラス: {type(self.model)}")
                print(f"モデル属性一覧: {dir(self.model)}")
                print(f"config属性: {self.model.config}")
                if hasattr(self.model.config, "text_config"):
                    print(f"text_config属性: {self.model.config.text_config}")
                    print(f"text_config.hidden_size: {self.model.config.text_config.hidden_size}")
                
                # 埋め込み層の構造を確認
                if hasattr(self.model, "get_input_embeddings"):
                    print(f"入力埋め込み層: {self.model.get_input_embeddings()}")
                if hasattr(self.model, "get_output_embeddings"):
                    print(f"出力埋め込み層: {self.model.get_output_embeddings()}")
                
                # Mllamaモデルの構造を分析してトークナイザに合わせてリサイズ
                print("Mllamaモデルの構造を分析中...")
                
                # 新しい語彙サイズを取得
                new_vocab_size = len(self.processor.tokenizer)
                print(f"リサイズ前: トークナイザサイズ={new_vocab_size}")
                
                try:
                    # 入力埋め込み層のリサイズ
                    print(f"トークナイザサイズに合わせてモデルをリサイズします: {new_vocab_size}")
                    self.model.resize_token_embeddings(new_vocab_size)
                    print(f"モデルを{self.device}に移動します")
                    self.model = self.model.to(self.device)
                except Exception as resize_error:
                    print(f"埋め込み層のリサイズ中にエラーが発生しました: {str(resize_error)}")
                    print("警告: モデルは標準のトークナイザサイズのままです")
                    self.model = self.model.to(self.device)
            
            except Exception as e:
                print(f"モデル初期化エラー: {str(e)}")
                print("AutoModelForCausalLMにフォールバックします")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                    ignore_mismatched_sizes=True,  # サイズ不一致を無視
                ).to(self.device)
                print("代わりにAutoModelForCausalLMを使用します")
                
                # 標準的なモデルのリサイズ
                try:
                    print("標準的なリサイズを試みます")
                    self.model.resize_token_embeddings(len(self.processor.tokenizer))
                except Exception as resize_error:
                    print(f"標準的なリサイズに失敗: {str(resize_error)}")
        
        print(f"モデルのデバイス: {self.device}")
        print(f"トークナイザの最終語彙サイズ: {len(self.processor.tokenizer)}")
        
        # マスキングするトークンIDを保存
        # 画像トークンなど、損失計算や生成から除外するトークンを設定
        self.special_token_ids = []
        # 画像トークンを特定（もし存在すれば）
        image_token = "<|image|>"
        if image_token in self.processor.tokenizer.get_vocab():
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
            self.special_token_ids.append(image_token_id)
            print(f"画像トークン {image_token} (ID: {image_token_id}) を特殊トークンとして登録しました")
        
        # PADトークンの追加と登録（必要な場合）
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            print(f"PADトークンをEOSトークン ({self.processor.tokenizer.eos_token}) に設定しました")
        
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            self.special_token_ids.append(pad_token_id)
            print(f"PADトークン (ID: {pad_token_id}) を特殊トークンとして登録しました")
        
        print(f"損失計算から除外される特殊トークンIDs: {self.special_token_ids}")

    def initialize_lisa_modules(self, sam_checkpoint=None):
        """LISAの追加モジュールを初期化
        
        Args:
            sam_checkpoint: SAMモデルのチェックポイントパス（任意）
        """
        print("\n=== LISAモジュールの初期化 ===")
        
        # デフォルトのSAM画像サイズを設定（まだ設定されていない場合）
        if not hasattr(self, "sam_image_size"):
            self.sam_image_size = 1024
            print(f"sam_image_sizeを{self.sam_image_size}に設定しました")
        
        # モデルのLLM次元を取得
        print("LLMの隠れ層次元を取得します...")
        try:
            config = getattr(self.model, "config", None)
            llm_hidden_size = None
            
            # configがMllamaConfigの場合
            if hasattr(config, "text_config"):
                print("Mllamaモデル構造を検出: config.text_configが存在します")
                text_config = config.text_config
                if hasattr(text_config, "hidden_size"):
                    llm_hidden_size = text_config.hidden_size
                    print(f"Mllamaのtext_config.hidden_size: {llm_hidden_size}")
            # 通常のLLMの場合
            elif hasattr(config, "hidden_size"):
                llm_hidden_size = config.hidden_size
                print(f"標準のconfig.hidden_size: {llm_hidden_size}")
            # configが辞書の場合
            elif isinstance(config, dict):
                if "text_config" in config and "hidden_size" in config["text_config"]:
                    llm_hidden_size = config["text_config"]["hidden_size"]
                    print(f"config辞書からtext_config.hidden_size: {llm_hidden_size}")
                elif "hidden_size" in config:
                    llm_hidden_size = config["hidden_size"]
                    print(f"config辞書からhidden_size: {llm_hidden_size}")
            
            # hidden_sizeが取得できなかった場合
            if llm_hidden_size is None:
                print("警告: モデルからhidden_sizeを取得できませんでした")
                # モデル構造を詳細に分析して手がかりを探す
                print("モデル構造を分析して次元を特定します...")
                
                if hasattr(self.model, "text_model") and hasattr(self.model.text_model, "config"):
                    text_model_config = self.model.text_model.config
                    if hasattr(text_model_config, "hidden_size"):
                        llm_hidden_size = text_model_config.hidden_size
                        print(f"text_model.config.hidden_size: {llm_hidden_size}")
                
                # それでも見つからない場合はデフォルト値を使用
                if llm_hidden_size is None:
                    llm_hidden_size = 4096  # デフォルト値
                    print(f"隠れ層次元が特定できないためデフォルト値を使用: {llm_hidden_size}")
            
            print(f"LLMの隠れ層次元: {llm_hidden_size}")
            
        except Exception as e:
            print(f"hidden_size取得中にエラーが発生: {str(e)}")
            llm_hidden_size = 4096  # エラー時のデフォルト値
            print(f"エラーによりデフォルトの隠れ層次元を使用: {llm_hidden_size}")
        
        # SAMモデルを初期化（以下の部分は変更なし）
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
                
                # SAMモデルが初期化された後にtransformを設定
                self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
                print(f"SAM画像リサイズ変換を初期化しました（サイズ: {self.sam.image_encoder.img_size}）")
                
                # テキスト->SAMプロンプト投影（Llama 3.2 Visionの隠れ状態次元から256次元へ）
                # すでに取得したllm_hidden_sizeを使用
                print(f"セグメンテーション投影を初期化: {llm_hidden_size} -> 256")
                self.seg_projection = nn.Linear(llm_hidden_size, 256)

                # セグメンテーション投影もGPUに移動
                self.seg_projection.to(self.device, self.dtype)

                # 重みを初期化（Kaiming初期化）
                nn.init.kaiming_normal_(
                    self.seg_projection.weight, nonlinearity='relu')
                    
            except Exception as e:
                print(f"SAMの初期化中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                # SAMの初期化に失敗した場合も、デフォルトのサイズでtransformを初期化
                self.transform = ResizeLongestSide(self.sam_image_size)
                print(f"デフォルトサイズでSAM画像リサイズ変換を初期化しました（サイズ: {self.sam_image_size}）")
                self.sam = None
        else:
            # SAMチェックポイントが指定されていない場合も、デフォルトのサイズでtransformを初期化
            self.transform = ResizeLongestSide(self.sam_image_size)
            print(f"SAMモデルなしでリサイズ変換のみ初期化しました（サイズ: {self.sam_image_size}）")

    def transform_image(self):
        """
        SAMモデル用の画像変換オブジェクトを作成します。

        Returns:
            transform: ResizeLongestSideオブジェクト
        """
        # SAMのデフォルト画像サイズが設定されていない場合は初期化
        if not hasattr(self, 'sam_image_size'):
            self.sam_image_size = 1024
            print(f"transform_image: sam_image_sizeが設定されていなかったので、デフォルト値({self.sam_image_size})を設定しました。")
            
        # SAM用のリサイズ変換を作成
        transform = ResizeLongestSide(target_length=self.sam_image_size)
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
                                        # 出力画像サイズとしてsam_image_sizeを使用
                                        masks.append(np.zeros((self.sam_image_size, self.sam_image_size), dtype=np.uint8))
                                        continue  # 次のトークンへ

                                # 隠れ状態を取得
                                # 最後の層から<seg>トークンの最後の隠れ状態を抽出
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                    hidden_states = outputs.hidden_states[-1]
                                    seg_hidden_state = hidden_states[0, -1]  # バッチ0、最後の位置
                                    
                                    print(f"<seg>トークンの隠れ状態: 形状={seg_hidden_state.shape}")
                                    
                                    # SAMの予測に使用する形式にプロジェクション
                                    prompt_embedding = self.seg_projection(seg_hidden_state)
                                    
                                    # GPUのデータ型に合わせる（Half精度で動作している場合はHalfに合わせる）
                                    device_dtype = next(self.sam.parameters()).dtype
                                    prompt_embedding = prompt_embedding.to(dtype=device_dtype)
                                    
                                    print(f"プロンプト埋め込みのデータ型: {prompt_embedding.dtype}, SAMモデルのデータ型: {device_dtype}")
                                    
                                    # SAMへの入力を準備
                                    # 画像特徴マップがdeviceと一致していることを確認
                                    sam_image_embedding = sam_image_embedding.to(prompt_embedding.device)
                                    
                                    try:
                                        # プロンプト埋め込みの形状調整
                                        # SAMモデルが期待する形式に変換 - [1, 1, 256]の形状にする
                                        sparse_embeddings = prompt_embedding.unsqueeze(0).unsqueeze(0)
                                        print(f"SAMへの入力 - sparse_embeddings: 形状={sparse_embeddings.shape}")
                                        
                                        # 空のdense_prompt_embeddingsを作成（Noneではなく空のテンソルを使用）
                                        # [B, C, H, W]形式のゼロテンソルを作成
                                        # 画像埋め込みから適切な形状を取得
                                        b, c, h, w = sam_image_embedding.shape
                                        dense_prompt_embeddings = torch.zeros(
                                            (b, c, h, w), 
                                            device=sam_image_embedding.device, 
                                            dtype=sam_image_embedding.dtype
                                        )
                                        
                                        # SAMモデルでマスクを予測
                                        masks_predictions, scores, logits = self.sam.mask_decoder(
                                            image_embeddings=sam_image_embedding,
                                            image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings=sparse_embeddings,
                                            dense_prompt_embeddings=dense_prompt_embeddings,
                                            multimask_output=False,
                                        )
                                        
                                        # マスク予測結果がNaNまたはInfを含んでいないか確認
                                        if torch.isnan(masks_predictions).any() or torch.isinf(masks_predictions).any():
                                            print("警告: マスク予測にNaNまたはInf値が含まれています。0に置き換えます。")
                                            masks_predictions = torch.nan_to_num(masks_predictions, nan=0.0, posinf=1.0, neginf=0.0)
                                        
                                        # マスクをリサイズして画像の元のサイズに合わせる
                                        mask = F.interpolate(
                                            masks_predictions,
                                            size=(self.sam_image_size, self.sam_image_size),
                                            mode="bilinear",
                                            align_corners=False,
                                        )
                                        
                                        # マスクを2値化
                                        mask = (mask > 0).float().cpu().numpy()
                                        
                                        # マスク情報をリストに追加 - ただし辞書ではなく直接マスク配列を追加
                                        masks.append(mask[0, 0])  # 最初のバッチ、最初のマスクのみ使用
                                        print(f"マスク生成成功: 形状={mask[0, 0].shape}, 型={type(mask[0, 0])}")
                                    
                                    except Exception as mask_error:
                                        print(f"マスク生成中にエラー: {str(mask_error)}")
                                        # エラーのトレース表示
                                        import traceback
                                        traceback.print_exc()
                                        # ダミーのマスクを追加 - ただし辞書ではなく直接マスク配列を追加
                                        dummy_mask = np.zeros((self.sam_image_size, self.sam_image_size), dtype=np.uint8)
                                        masks.append(dummy_mask)
                                        print(f"ダミーマスク生成: 形状={dummy_mask.shape}")
                                
                                else:
                                    print("警告: hidden_statesが取得できませんでした")
                                    # ダミーのマスクを追加 - 辞書ではなく直接マスク配列を追加
                                    dummy_mask = np.zeros((self.sam_image_size, self.sam_image_size), dtype=np.uint8)
                                    masks.append(dummy_mask)
                                    print(f"hidden_statesなしのダミーマスク生成: 形状={dummy_mask.shape}")
                        
                        except Exception as seg_error:
                            print(f"セグメンテーショントークン処理中にエラー: {str(seg_error)}")
                            masks.append(np.zeros((self.sam_image_size, self.sam_image_size), dtype=np.uint8))
                            
                except Exception as token_error:
                    print(f"トークン処理中にエラー: {str(token_error)}")
                    # エラーの詳細を表示
                    import traceback
                    traceback.print_exc()
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

        # マスクの形式を確認して、正しい形式で返す
        print(f"生成されたマスク数: {len(masks)}")
        for i, mask in enumerate(masks):
            print(f"マスク {i+1} の形状: {mask.shape}, 型: {type(mask)}")

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

    def train_step(self, batch, optimizer):
        """
        学習ステップを実行する関数
        """
        # 入力データの準備
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 特殊トークンを損失計算から除外（-100は無視される）
        if hasattr(self, "special_token_ids") and self.special_token_ids:
            for token_id in self.special_token_ids:
                labels[labels == token_id] = -100
        
        # 画像データがある場合
        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"].to(self.device)
            
            # アスペクト比情報（画像のタイル処理用）があれば取得
            aspect_ratio_ids = batch.get("aspect_ratio_ids", None)
            if aspect_ratio_ids is not None:
                aspect_ratio_ids = aspect_ratio_ids.to(self.device)
            
            aspect_ratio_mask = batch.get("aspect_ratio_mask", None)
            if aspect_ratio_mask is not None:
                aspect_ratio_mask = aspect_ratio_mask.to(self.device)
            
            # DeepSpeedの場合
            if self.use_deepspeed:
                # DeepSpeedモデルのforwardを呼び出し
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values,
                    aspect_ratio_ids=aspect_ratio_ids,
                    aspect_ratio_mask=aspect_ratio_mask,
                    return_dict=True
                )
            else:
                # 通常のPyTorchモデル
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values,
                    aspect_ratio_ids=aspect_ratio_ids,
                    aspect_ratio_mask=aspect_ratio_mask,
                    return_dict=True
                )
        else:
            # テキストのみの場合
            if self.use_deepspeed:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
        
        # 損失の取得
        loss = outputs.loss
        
        # DeepSpeedの場合はbackward()をDeepSpeedが処理
        if self.use_deepspeed:
            self.model.backward(loss)
            self.model.step()
        else:
            # 通常のPyTorch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return loss.item()


def build_sam_vit_h(checkpoint=None):
    """
    SAM ViT-H モデルを構築
    """
    from .segment_anything import sam_model_registry
    
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    if checkpoint is not None:
        print(f"SAMチェックポイントから重みを読み込みました: {checkpoint}")
    else:
        print("SAMチェックポイントが指定されていません - 学習済み重みなしで初期化します")
    return sam
