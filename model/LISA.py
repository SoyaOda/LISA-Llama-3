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
import torch.distributed as dist
import re
import traceback
import io
import base64


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
        image_size=1024,  # image_sizeをコンストラクタの引数に追加
        infer=True  # 推論モードかトレーニングモードかを指定するフラグ
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
            infer: 推論モードかどうか (デフォルト: True)
        """
        super().__init__()
        
        # 基本パラメータの設定
        self.seg_token = seg_token
        self.max_batch_size = max_batch_size
        self.use_deepspeed = use_deepspeed
        self.ds_config = ds_config
        self.local_rank = local_rank
        self.sam_image_size = image_size  # sam_image_sizeを初期化
        self.transform = None
        self.infer = infer  # 推論モードかどうかのフラグ
        
        # デバイスの設定
        if device is None:
            if torch.cuda.is_available():
                if use_deepspeed and local_rank >= 0:
                    self.device = f"cuda:{local_rank}"
                else:
                    self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"モデルのデバイスを設定: {self.device}")
        
        # データ型の設定（GPUが利用可能であればfloat16、そうでなければfloat32）
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.dtype = self.torch_dtype  # 別名
        
        # モデル初期化
        self.model = None
        self.processor = None
        self.initialize_llama(model_path)
        
        # SAMモデルがある場合は初期化
        self.sam = None
        self.initialize_lisa_modules(sam_checkpoint)

    def initialize_llama(self, model_path):
        """
        Llama 3.2 Vision モデルを初期化
        """
        print(f"プロセッサーとモデルをロードしています: {model_path}")
        
        # AutoProcessorの場合はtrust_remote_codeを使用
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # <seg>トークンをトークナイザに追加 - 先にトークナイザに追加（埋め込み拡張前）
        print(f"トークナイザにセグメンテーショントークンを追加します: {self.seg_token}")
        num_added_tokens = self.processor.tokenizer.add_tokens([self.seg_token])
        print(f"追加されたトークン数: {num_added_tokens}")
        
        # セグメンテーショントークンのIDを取得
        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
        print(f"セグメンテーショントークンID: {self.seg_token_idx}")
        print(f"トークナイザの語彙サイズ: {len(self.processor.tokenizer)}")
        
        # 両方のモデルクラスを事前にインポート
        from transformers import AutoModelForCausalLM
        try:
            from transformers import MllamaForConditionalGeneration
            model_class = MllamaForConditionalGeneration
            print("MllamaForConditionalGenerationを使用します")
        except Exception as e:
            print(f"MllamaForConditionalGenerationの使用に失敗: {str(e)}")
            model_class = AutoModelForCausalLM
            print("代わりにAutoModelForCausalLMを使用します")
        
        # DeepSpeedを使用する場合
        if self.use_deepspeed:
            print("DeepSpeedエンジンを初期化しています...")
            
            # world_sizeを取得（DeepSpeedの場合）
            if dist.is_initialized():
                world_size = dist.get_world_size()
                local_rank = dist.get_rank()
            else:
                world_size = 1
                local_rank = 0
                
            # マイクロバッチサイズとバッチサイズを設定
            micro_batch_size = 1
            grad_accum_steps = 1
            total_batch_size = micro_batch_size * grad_accum_steps * world_size
            
            print(f"DeepSpeed設定: world_size={world_size}, local_rank={local_rank}, micro_batch={micro_batch_size}, grad_accum={grad_accum_steps}")
            print(f"計算された合計バッチサイズ: {total_batch_size}")
            
            # 設定ファイルの処理
            ds_config_dict = None
            
            # 設定ファイルが指定されている場合は読み込む
            if isinstance(self.ds_config, str) and os.path.exists(self.ds_config):
                try:
                    with open(self.ds_config, 'r') as f:
                        ds_config_dict = json.load(f)
                    print(f"既存のDeepSpeed設定ファイルを読み込みました: {self.ds_config}")
                except Exception as e:
                    print(f"設定ファイル読み込みエラー: {str(e)}")
                    ds_config_dict = None
            
            # 設定ファイルが読み込めなかった場合はデフォルト設定を使用
            if ds_config_dict is None:
                # 推論モード用の設定
                if getattr(self, 'infer', True):  # デフォルトは推論モード
                    print("推論モード用のDeepSpeed設定を使用します")
                    ds_config_dict = {
                        "fp16": {
                            "enabled": True
                        },
                        "bf16": {
                            "enabled": False
                        },
                        "zero_optimization": {
                            "stage": 0  # 推論時はステージ0を使用
                        },
                        "train_batch_size": total_batch_size,
                        "train_micro_batch_size_per_gpu": micro_batch_size,
                        "steps_per_print": 50,
                        "wall_clock_breakdown": False
                    }
                else:
                    # 学習モード用の設定
                    print("学習モード用のDeepSpeed設定を使用します")
                    ds_config_dict = {
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
                            "stage": 2,
                            "offload_optimizer": {
                                "device": "cpu",
                                "pin_memory": True
                            },
                            "contiguous_gradients": True,
                            "overlap_comm": True
                        },
                        "gradient_accumulation_steps": grad_accum_steps,
                        "gradient_clipping": 1.0,
                        "steps_per_print": 50,
                        "train_batch_size": total_batch_size,
                        "train_micro_batch_size_per_gpu": micro_batch_size,
                        "wall_clock_breakdown": False
                    }
                print("デフォルトのDeepSpeed設定を使用します")
            else:
                # 既存の設定にworld_sizeに基づくバッチサイズを設定
                ds_config_dict["train_batch_size"] = total_batch_size
                ds_config_dict["train_micro_batch_size_per_gpu"] = micro_batch_size
                # 推論モードの場合はZeRO-stageを0に設定
                if getattr(self, 'infer', True) and "zero_optimization" in ds_config_dict:
                    ds_config_dict["zero_optimization"]["stage"] = 0
                    print("推論モード用にZeRO-stageを0に設定しました")
                print("既存の設定ファイルを修正しました")
            
            # プロセス固有の一時的な設定ファイルに書き出し
            temp_config_path = f"temp_ds_config_rank{local_rank}.json"
            with open(temp_config_path, "w") as f:
                json.dump(ds_config_dict, f, indent=2)
            
            self.ds_config = temp_config_path
            print(f"DeepSpeed設定ファイルを作成: {self.ds_config}")
            
            # モデルの初期化（DeepSpeed初期化前）
            print("モデルを一時的に初期化してトークナイザと同期します")
            
            # モデルクラスに応じて適切な引数を設定
            if model_class == AutoModelForCausalLM:
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                }
            else:
                # MllamaForConditionalGenerationの場合はtrust_remote_codeを使用しない
                model_kwargs = {
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                }
            
            # 一時モデルの初期化
            temp_model = model_class.from_pretrained(
                model_path,
                **model_kwargs
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
                
                # 出力埋め込み層の手動リサイズ（必要な場合）
                if hasattr(temp_model, "get_output_embeddings") and temp_model.get_output_embeddings() is not None:
                    old_embeddings = temp_model.get_output_embeddings()
                    print(f"出力埋め込み層を取得: {old_embeddings}")
                
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
            # プロセス固有の一時ディレクトリを使用
            temp_save_dir = f"./temp_resized_model_rank{local_rank}"
            os.makedirs(temp_save_dir, exist_ok=True)
            temp_model.save_pretrained(temp_save_dir)
            del temp_model
            torch.cuda.empty_cache()
            
            # 保存したリサイズ済みモデルを使ってDeepSpeedを初期化
            print(f"リサイズしたモデルを使用してDeepSpeedを初期化: {temp_save_dir}")
            try:
                # モデルクラスに応じて適切な引数を設定
                if model_class == AutoModelForCausalLM:
                    model_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                        "ignore_mismatched_sizes": True,
                    }
                else:
                    # MllamaForConditionalGenerationの場合はtrust_remote_codeを使用しない
                    model_kwargs = {
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                        "ignore_mismatched_sizes": True,
                    }
                
                # 推論時用のパラメータを追加：オプティマイザとして空のダミーオプティマイザを提供
                model = model_class.from_pretrained(
                    temp_save_dir,
                    **model_kwargs
                )
                
                # 推論モードの場合はZeROなしで初期化
                if getattr(self, 'infer', True):
                    print("推論モードでDeepSpeedを初期化します")
                    self.model, _, _, _ = deepspeed.initialize(
                        model=model,
                        config=self.ds_config,
                        model_parameters=None
                    )
                else:
                    # 学習モードの場合はダミーオプティマイザを使用
                    print("学習モードでDeepSpeedを初期化します（ダミーオプティマイザ）")
                    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
                    self.model, _, _, _ = deepspeed.initialize(
                        model=model,
                        optimizer=dummy_optimizer,
                        config=self.ds_config,
                        model_parameters=None
                    )
                    
                print("DeepSpeedの初期化が成功しました")
            except Exception as e:
                print(f"DeepSpeed初期化エラー: {str(e)}")
                print("標準モデルにフォールバックします")
                
                # エラー時のフォールバック: 標準モデルを使用
                self.model = model_class.from_pretrained(
                    temp_save_dir,
                    **model_kwargs
                ).to(self.device)
                print("標準モデルを使用します")
            
            # 一時チェックポイントを削除
            try:
                # 他のプロセスが同時にアクセスするのを防ぐためにバリアを追加（分散環境の場合）
                if dist.is_initialized():
                    dist.barrier()
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
                # モデルクラスに応じて適切な引数を設定
                if model_class == AutoModelForCausalLM:
                    model_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": torch.float16 if "cuda" in self.device else torch.float32,
                        "ignore_mismatched_sizes": True,
                    }
                else:
                    # MllamaForConditionalGenerationの場合はtrust_remote_codeを使用しない
                    model_kwargs = {
                        "torch_dtype": torch.float16 if "cuda" in self.device else torch.float32,
                        "ignore_mismatched_sizes": True,
                    }
                
                # まず標準のモデルをロード
                self.model = model_class.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                print(f"{model_class.__name__}を使用します")
                
                # デバッグ: モデル構造の詳細を出力
                print(f"\n===== {model_class.__name__} モデル構造 =====")
                print(f"モデルクラス: {type(self.model)}")
                print(f"config属性: {self.model.config}")
                if hasattr(self.model.config, "text_config"):
                    print(f"text_config属性: {self.model.config.text_config}")
                    print(f"text_config.hidden_size: {self.model.config.text_config.hidden_size}")
                
                # 埋め込み層の構造を確認
                if hasattr(self.model, "get_input_embeddings"):
                    print(f"入力埋め込み層: {self.model.get_input_embeddings()}")
                if hasattr(self.model, "get_output_embeddings"):
                    print(f"出力埋め込み層: {self.model.get_output_embeddings()}")
                
                # モデルの構造を分析してトークナイザに合わせてリサイズ
                print("モデルの構造を分析中...")
                
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
                
                # AutoModelForCausalLMの場合の引数
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if "cuda" in self.device else torch.float32,
                    "ignore_mismatched_sizes": True,
                }
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
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
                
                # オリジナルサイズを保存（高さ、幅の順）
                self.original_image_size = (image.height, image.width)
                print(f"オリジナル画像サイズ: {self.original_image_size}")

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
        do_sample=True
    ):
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
            if self.use_deepspeed:
                # DeepSpeedモデルの場合、moduleからデバイスを取得
                if hasattr(self.model, 'module'):
                    device = next(self.model.module.parameters()).device
                else:
                    device = next(self.model.parameters()).device
            else:
                # 通常のモデルの場合
                device = next(self.model.parameters()).device
                
            # デバイスがCPUの場合、self.deviceが設定されていればそれを使用
            if device.type == 'cpu' and 'cuda' in self.device:
                print(f"警告: モデルがCPUに配置されています。{self.device}に移動します。")
                device = torch.device(self.device)
                
            print(f"デバイス: {device}")
            
            # 現在のGPUメモリ使用状況を表示
            if device.type == 'cuda':
                print(f"GPUメモリ使用状況（推論前）: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            
            # モデルのデバイスが自身のデバイス属性と一致するか確認
            print(f"モデルのデバイス: {device if device else self.device}")
            
            # SAMで使用する画像処理
            # 画像が有効な形式かチェック
            image_pil = check_image(image)

            # 画像サイズを記録（後でマスクのリサイズに使用）
            self.original_image_size = (image_pil.height, image_pil.width)
            print(f"オリジナル画像サイズ: {self.original_image_size}")

            # SAMモデル用の画像埋め込みを生成
            sam_image_embedding = self.preprocess_sam_image(image_pil)
            print(f"SAM画像埋め込みのシェイプ: {sam_image_embedding.shape}")

            # プロンプトの言語を検出
            is_japanese = any(ord(c) > 127 for c in prompt)
            is_english = not is_japanese
            
            # 日本語プロンプトの場合、明示的に日本語で回答するよう指示を追加
            if is_japanese:
                if "<seg>" in prompt:
                    # <seg>トークンの位置を保持
                    seg_pos = prompt.find("<seg>")
                    # <seg>の前にだけ指示を追加
                    prompt = prompt[:seg_pos] + "必ず日本語で詳しく答えてください。 " + prompt[seg_pos:]
                else:
                    prompt = prompt + " 必ず日本語で詳しく答えてください。"
                
                system_content = "あなたは画像理解に優れた高性能AIアシスタントです。ユーザーの質問に対して、画像の内容を詳しく分析し、日本語で明確かつ詳細に回答してください。特に人物や物体の特徴、位置関係、色彩などを具体的に説明することを心がけてください。"
            else:
                system_content = "You are a helpful vision-language assistant that understands images and text. Analyze the image carefully and provide clear, detailed explanations."
            
            # Llama 3.2 Visionの推奨プロンプト形式を使用
            try:
                # ユーザーメッセージのみの形式
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "source": {"data": "data:image/jpeg;base64," + image_to_base64(image_pil), "width": image_pil.width, "height": image_pil.height}},
                        {"type": "text", "text": prompt}
                    ]}
                ]
                
                # モデルに合ったチャットテンプレートを適用
                input_text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                print(f"チャットテンプレート適用状況: 標準テンプレート使用")
            except Exception as e:
                print(f"標準テンプレート適用エラー: {e}")
                # フォールバック：直接テンプレートを構築
                input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                print("フォールバック：手動テンプレートを適用しました")
                
            print(f"チャットテンプレート適用後: {input_text[:100]}...")

            try:
                # モデルに画像とテキストを入力
                with torch.inference_mode():
                    inputs = self.processor(
                        images=image_pil,
                        text=input_text,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # テンソルをモデルのデバイスに移動
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(device)
                    
                    # 入力テンソルの形状確認
                    if 'pixel_values' in inputs:
                        print(f"pixel_values: 形状={inputs['pixel_values'].shape}")
                
                # GPUメモリ使用状況をデバッグ出力（推論前）
                if torch.cuda.is_available() and device.type == 'cuda':
                    print(f"生成前のGPUメモリ: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

                # 言語に応じた最適な生成パラメータを設定
                if is_japanese:
                    # 日本語生成の最適パラメータ
                    print("日本語プロンプトを検出: 日本語生成パラメータを適用")
                    generation_kwargs = {
                        'do_sample': True,
                        'temperature': 0.2,     # 低温設定
                        'top_p': 0.9,
                        'repetition_penalty': 1.0,  # 繰り返しペナルティは最小に
                        'max_new_tokens': max_new_tokens,
                        'num_beams': 1,         # ビームサーチはオフ
                        'use_cache': True,
                    }
                else:
                    # 英語生成の最適パラメータ
                    print("英語プロンプトを検出: 英語生成パラメータを適用")
                    generation_kwargs = {
                        'do_sample': True,
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'repetition_penalty': 1.0,
                        'max_new_tokens': max_new_tokens,
                        'num_beams': 1,
                        'use_cache': True,
                    }

                # ユーザー指定のパラメータで上書き（指定がある場合のみ）
                if temperature is not None and temperature > 0:
                    generation_kwargs['temperature'] = temperature
                if top_p is not None and 0 < top_p <= 1.0:
                    generation_kwargs['top_p'] = top_p
                if repetition_penalty is not None and repetition_penalty > 0:
                    generation_kwargs['repetition_penalty'] = repetition_penalty
                if do_sample is not None:
                    generation_kwargs['do_sample'] = do_sample
                if top_k is not None and top_k > 0:
                    generation_kwargs['top_k'] = top_k
                
                # 重要なパラメータを出力
                print(f"生成パラメータ: temperature={generation_kwargs.get('temperature', 'N/A')}, " +
                      f"top_p={generation_kwargs.get('top_p', 'N/A')}, " +
                      f"repetition_penalty={generation_kwargs.get('repetition_penalty', 'N/A')}, " +
                      f"do_sample={generation_kwargs.get('do_sample', 'N/A')}, " +
                      f"num_beams={generation_kwargs.get('num_beams', 'N/A')}")

                # 生成開始
                generate_ids = None

                # DeepSpeedモデルの場合
                if self.use_deepspeed and hasattr(self.model, 'module'):
                    print("DeepSpeedモデルを使用して生成します...")
                    
                    try:
                        # モデルで重みの共有を確認し、必要に応じて修正
                        if hasattr(self.model.module, "tie_weights") and callable(getattr(self.model.module, "tie_weights")):
                            self.model.module.tie_weights()
                            print("モデルの重みを結合しました")
                        
                        # モジュールの generate メソッドを呼び出す
                        if hasattr(self.model.module, 'generate'):
                            print("model.module.generate()を使用します")
                            with torch.no_grad():
                                # deepspeedオプティマイザ状態をクリア
                                if hasattr(self.model, "optimizer") and self.model.optimizer is not None:
                                    self.model.optimizer.zero_grad()
                                
                                # ダミー推論を行ってDeepSpeedの内部状態を初期化（問題報告への対応）
                                _ = self.model.module.forward(**{k: v[:1] for k, v in inputs.items()})
                                
                                # 本推論実行
                                generate_ids = self.model.module.generate(
                                    **inputs,
                                    **generation_kwargs
                                )
                        else:
                            print("generate()メソッドが見つかりません")
                            raise ValueError("モデルにgenerate()メソッドがありません")
                    except Exception as e:
                        print(f"モデル生成時にエラー発生: {str(e)}")
                        print("詳細情報:", traceback.format_exc())
                        raise e
                
                # 標準のPyTorchモデルの場合
                else:
                    print("標準のPyTorchモデルを使用して生成します...")
                    
                    try:
                        # モデルで重みの共有を確認し、必要に応じて修正
                        if hasattr(self.model, "tie_weights") and callable(getattr(self.model, "tie_weights")):
                            self.model.tie_weights()
                            print("モデルの重みを結合しました")
                        
                        # モデルが.generateメソッドを持っているか確認
                        if hasattr(self.model, 'generate'):
                            print("model.generate()を使用します")
                            with torch.no_grad():
                                generate_ids = self.model.generate(
                                    **inputs,
                                    **generation_kwargs
                                )
                        else:
                            print("generate()メソッドが見つかりません")
                            raise ValueError("モデルにgenerate()メソッドがありません")
                    except Exception as e:
                        print(f"モデル生成時にエラー発生: {str(e)}")
                        print("詳細情報:", traceback.format_exc())
                        raise e
                
                # 生成されたトークンをデコード
                try:
                    # トークンIDから文字列にデコード
                    raw_text = self.processor.batch_decode(
                        generate_ids, 
                        skip_special_tokens=False,  # まず特殊トークンを保持してデコード
                        clean_up_tokenization_spaces=True
                    )[0]
                    
                    print(f"生のデコード結果（処理前）の長さ: {len(raw_text)}")
                    print(f"生のデコード結果（処理前）の一部: {raw_text[:150]}...")
                    
                    # アシスタント応答を抽出（正規表現を使用）
                    assistant_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)(?:<\|eot_id\|>|$)'
                    assistant_matches = re.findall(assistant_pattern, raw_text, re.DOTALL)
                    
                    if assistant_matches:
                        # 最後のアシスタント応答を使用
                        extracted_text = assistant_matches[-1].strip()
                        print(f"アシスタント応答が正常に抽出されました: {len(extracted_text)}文字")
                    else:
                        # 抽出失敗時の別パターン試行
                        alt_patterns = [
                            r'assistant\s*:(.*?)(?:user:|$)',
                            r'<assistant>(.*?)</assistant>',
                            r'\|\s*assistant\s*\|(.*?)(?:\||$)'
                        ]
                        
                        for pattern in alt_patterns:
                            matches = re.findall(pattern, raw_text, re.DOTALL)
                            if matches:
                                extracted_text = matches[-1].strip()
                                print(f"代替パターンでアシスタント応答を抽出: {len(extracted_text)}文字")
                                break
                        else:
                            # 入力以降の部分を抽出
                            print("パターンマッチ失敗: 入力以降のテキストを使用")
                            input_len = len(inputs["input_ids"][0])
                            response_ids = generate_ids[0][input_len:]
                            extracted_text = self.processor.decode(response_ids, skip_special_tokens=True)
                            print(f"代替方法で応答を抽出: {len(extracted_text)}文字")
                    
                    # セグメンテーショントークンが処理中に消えていたら復元
                    if self.seg_token not in extracted_text and raw_text.find(self.seg_token) != -1:
                        # セグメンテーショントークンを文末に追加
                        extracted_text = extracted_text.rstrip() + f" {self.seg_token}"
                        print(f"セグメンテーショントークン {self.seg_token} を復元しました")
                    
                    # 最終的なテキスト
                    generated_text = extracted_text
                    
                    # 生成テキストの情報を表示
                    print(f"生成されたテキスト: 長さ={len(generated_text)}文字")
                    if len(generated_text) > 100:
                        print(f"生成テキストの先頭部分: {generated_text[:100]}...")
                    else:
                        print(f"生成テキスト全体: {generated_text}")
                        
                except Exception as decode_error:
                    print(f"テキストデコード中にエラー: {str(decode_error)}")
                    traceback_str = traceback.format_exc()
                    print(f"詳細なトレースバック:\n{traceback_str}")
                    
                    # エラー回復：最低限のテキストを提供
                    generated_text = f"テキスト生成エラーが発生しました。{self.seg_token}"
                
                # セグメンテーショントークンの位置を検索
                try:
                    # セグメンテーショントークンのIDを取得
                    seg_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.seg_token)
                    print(f"<seg>トークンID: {seg_token_id}")
                    
                    # 生成テキスト内の<seg>トークン位置を検索
                    seg_positions = []
                    for batch_idx in range(generate_ids.shape[0]):
                        positions = torch.where(generate_ids[batch_idx] == seg_token_id)[0]
                        for pos in positions:
                            seg_positions.append((batch_idx, pos.item()))
                    
                    print(f"<seg>トークン位置: {seg_positions}")
                    
                    # <seg>トークンが見つかった場合、マスクを生成
                    if seg_positions:
                        for batch_idx, pos in seg_positions:
                            # この位置までのシーケンスを抽出
                            input_ids_segment = generate_ids[batch_idx, :pos+1].unsqueeze(0)
                            attention_mask_segment = torch.ones_like(input_ids_segment)
                            
                            # フォワードパスを実行して隠れ状態を取得
                            with torch.no_grad():
                                # モデルがモジュール（DeepSpeed）かどうかチェック
                                model_for_inference = self.model.module if hasattr(self.model, 'module') else self.model
                                
                                # 入力テンソルの準備
                                forward_inputs = {
                                    'input_ids': input_ids_segment.to(device),
                                    'attention_mask': attention_mask_segment.to(device),
                                }
                                
                                # pixel_valuesを追加（必要な場合）
                                if 'pixel_values' in inputs:
                                    forward_inputs['pixel_values'] = inputs['pixel_values'].to(device)
                                
                                # その他の必要な入力があれば追加
                                for k, v in inputs.items():
                                    if k not in forward_inputs and k != 'input_ids' and k != 'attention_mask':
                                        forward_inputs[k] = v.to(device)
                                
                                # 出力にhidden_statesを含めるよう設定
                                forward_inputs['output_hidden_states'] = True
                                forward_inputs['return_dict'] = True
                                
                                # モデルの順伝播を実行
                                try:
                                    outputs = model_for_inference(**forward_inputs)
                                    print("GPUでの隠れ状態取得に成功しました")
                                except Exception as e:
                                    print(f"GPUでの隠れ状態取得中にエラー: {str(e)}")
                                    print("より詳細なエラー情報:", traceback.format_exc())
                                    raise e
                                
                                # 隠れ状態を取得
                                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                                    # 最後の層の最後の位置（<seg>トークン）の隠れベクトルを取得
                                    last_hidden_states = outputs.hidden_states[-1]
                                    seg_hidden_state = last_hidden_states[0, -1]
                                    
                                    print(f"<seg>トークンの隠れ状態: 形状={seg_hidden_state.shape}")
                                    
                                    # セグメンテーション投影を適用
                                    prompt_embedding = self.seg_projection(seg_hidden_state)
                                    
                                    # NaN/Infチェック
                                    if torch.isnan(prompt_embedding).any() or torch.isinf(prompt_embedding).any():
                                        print("警告: 埋め込みにNaNまたはInf値があります。置換します。")
                                        prompt_embedding = torch.nan_to_num(prompt_embedding)
                                    
                                    # SAMモデルのデータ型に変換
                                    prompt_embedding = prompt_embedding.to(
                                        device=device,
                                        dtype=next(self.sam.parameters()).dtype
                                    )
                                    
                                    # SAM用の画像埋め込みを取得
                                    with torch.no_grad():
                                        # SAMの画像埋め込みをデバイスと型に合わせる
                                        sam_image_embedding = sam_image_embedding.to(
                                            device=prompt_embedding.device,
                                            dtype=prompt_embedding.dtype
                                        )
                                        
                                        # プロンプト埋め込みを適切な形状に変形
                                        sparse_embeddings = prompt_embedding.unsqueeze(0).unsqueeze(0)
                                        
                                        # 位置エンコーディングを取得
                                        image_pe = self.sam.prompt_encoder.get_dense_pe()
                                        image_pe = image_pe.to(
                                            device=prompt_embedding.device,
                                            dtype=prompt_embedding.dtype
                                        )
                                        
                                        # 位置エンコーディングを画像埋め込みに合わせてリサイズ
                                        if image_pe.shape[2:] != sam_image_embedding.shape[2:]:
                                            image_pe = F.interpolate(
                                                image_pe,
                                                size=sam_image_embedding.shape[2:],
                                                mode='bilinear',
                                                align_corners=False
                                            )
                                        
                                        # 空の密なプロンプト埋め込みを作成
                                        b, c, h, w = sam_image_embedding.shape
                                        dense_embeddings = torch.zeros(
                                            (b, c, h, w),
                                            device=sam_image_embedding.device,
                                            dtype=sam_image_embedding.dtype
                                        )
                                        
                                        # マスクデコーダを使用してマスクを予測
                                        try:
                                            mask_predictions, _ = self.sam.mask_decoder(
                                                image_embeddings=sam_image_embedding,
                                                image_pe=image_pe,
                                                sparse_prompt_embeddings=sparse_embeddings,
                                                dense_prompt_embeddings=dense_embeddings,
                                                multimask_output=False
                                            )
                                            
                                            print("マスク予測成功！")
                                            
                                            # マスクを元の画像サイズにリサイズ
                                            mask = F.interpolate(
                                                mask_predictions,
                                                size=self.original_image_size,
                                                mode='bilinear',
                                                align_corners=False
                                            )
                                            
                                            # シグモイドを適用して閾値処理
                                            mask = torch.sigmoid(mask) > 0.5
                                            
                                            # マスクをCPU、NumPy配列に変換
                                            mask_np = mask.cpu().numpy()[0, 0]
                                            masks.append(mask_np)
                                            
                                            print(f"マスク生成成功: 形状={mask_np.shape}")
                                            
                                        except Exception as mask_error:
                                            print(f"マスク生成中にエラー: {str(mask_error)}")
                                            print("詳細エラー情報:", traceback.format_exc())
                                            
                                            # ダミーマスクを作成（オリジナル画像サイズで）
                                            h, w = self.original_image_size
                                            dummy_mask = np.zeros((h, w), dtype=np.float32)
                                            masks.append(dummy_mask)
                                            
                                            print(f"代替ダミーマスクを生成: 形状={dummy_mask.shape}")
                                    
                                else:
                                    print("警告: モデル出力に隠れ状態が含まれていません")
                                    # ダミーマスクを作成
                                    h, w = self.original_image_size
                                    dummy_mask = np.zeros((h, w), dtype=np.float32)
                                    masks.append(dummy_mask)
                        
                        # マスクが生成されなかった場合のフォールバック
                        if not masks:
                            print("警告: マスクが生成されませんでした。ダミーマスクを作成します。")
                            h, w = self.original_image_size
                            dummy_mask = np.zeros((h, w), dtype=np.float32)
                            masks.append(dummy_mask)
                    
                    # セグメンテーショントークンが見つからなかった場合
                    else:
                        print("警告: セグメンテーショントークンが見つかりませんでした。")
                        print("テキスト内のセグメンテーショントークンを探索します...")
                        
                        if self.seg_token in generated_text:
                            print(f"テキスト内で{self.seg_token}を検出しました。ダミーマスクを生成します。")
                        else:
                            print(f"テキスト内に{self.seg_token}が見つかりませんでした。これは予期しない状態です。")
                            # セグメンテーショントークンを追加
                            generated_text += f" {self.seg_token}"
                        
                        # ダミーマスクを作成
                        h, w = self.original_image_size
                        dummy_mask = np.zeros((h, w), dtype=np.float32)
                        masks.append(dummy_mask)
                    
                except Exception as token_error:
                    print(f"トークン検索中にエラー: {str(token_error)}")
                    print("詳細エラー情報:", traceback.format_exc())
                    # エラー時のフォールバック
                    return {"masks": [], "text": generated_text}
            
            except Exception as process_error:
                print(f"入力処理中にエラー: {str(process_error)}")
                print("詳細エラー情報:", traceback.format_exc())
                return {"masks": [], "text": f"入力処理エラー: {str(process_error)}"}
            
            # 最終的な結果を返す
            print(f"生成されたテキスト(LISA内部): 長さ={len(generated_text)}文字, 先頭部分={generated_text[:50]}...")
            return {"masks": masks, "text": generated_text}
            
        except Exception as e:
            print(f"画像とテキストの処理中にエラーが発生しました: {str(e)}")
            traceback.print_exc()
            return {"masks": [], "text": f"画像処理エラー: {str(e)}"}

    @classmethod
    def from_pretrained(cls, model_path, sam_checkpoint=None, use_deepspeed=False, ds_config=None, local_rank=-1, infer=True, **kwargs):
        """
        事前学習済みモデルからLISAモデルを作成

        Args:
            model_path: モデルのパス
            sam_checkpoint: SAMチェックポイントのパス
            use_deepspeed: DeepSpeedを使用するかどうか
            ds_config: DeepSpeedの設定
            local_rank: ローカルランク（DeepSpeed用）
            infer: 推論モードかどうか (デフォルト: True)
            **kwargs: その他の引数
        """
        return cls(
            model_path=model_path,
            sam_checkpoint=sam_checkpoint,
            use_deepspeed=use_deepspeed,
            ds_config=ds_config,
            local_rank=local_rank,
            infer=infer,
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

def image_to_base64(image):
    """画像をbase64エンコードする補助関数"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

