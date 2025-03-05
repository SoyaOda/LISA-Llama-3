"""
Llama 3.2 Vision (Mllama) model implementation for LISA project.
This module integrates Llama 3.2 Vision with SAM for segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor, 
    MllamaForConditionalGeneration
)
from typing import List, Dict, Optional, Tuple, Any, Union

# 特殊トークンの定義
SEG_TOKEN = "<seg>"  # セグメンテーショントークン

class LisaLlama3MetaModel:
    """
    Llama 3.2 Vision (Mllama)とSAMを統合するための基本モデルクラス。
    """
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # SAMチェックポイントをkwargsから取得
        self.sam_checkpoint = kwargs.get("sam_checkpoint", None)
        if self.sam_checkpoint is None:
            print("警告: SAMチェックポイントが指定されていません。")
        
        # SAMモデルを初期化
        self.initialize_lisa_modules(config, **kwargs)
    
    def initialize_lisa_modules(self, config, **kwargs):
        """
        SAMモデルと射影層の初期化
        """
        from ..segment_anything import build_sam_vit_h
        
        # SAMモデルの読み込みと初期化
        if self.sam_checkpoint is not None:
            print(f"SAMチェックポイントを読み込みます: {self.sam_checkpoint}")
            self.sam = build_sam_vit_h(checkpoint=self.sam_checkpoint)
            self.sam.eval()
            for param in self.sam.parameters():
                param.requires_grad = False
        else:
            print("SAMチェックポイントが指定されていないため、SAMモデルは初期化されません")
            self.sam = None
            
        # SAMのMaskDecoderに送る前の射影層
        # Llama 3.2のtext_config.hidden_sizeからSAMのプロンプト次元(256)への変換
        # config.text_config.hidden_sizeは、Llama 3.2では通常4096
        self.seg_projection = nn.Linear(
            config.text_config.hidden_size, 
            256  # SAMのプロンプト次元
        )
        
        # 損失重みの設定
        self.seg_token_idx = None  # トークナイザーで設定される
        self.dice_loss_weight = getattr(config, "dice_loss_weight", 0.5)
        self.bce_loss_weight = getattr(config, "bce_loss_weight", 0.5)
        

class LisaLlama3Model(LisaLlama3MetaModel, MllamaForConditionalGeneration):
    """
    Llama 3.2 Vision (Mllama)モデルとSAMを組み合わせたLISAモデルクラス
    """
    def __init__(
        self,
        config,
        **kwargs,
    ):
        MllamaForConditionalGeneration.__init__(self, config)
        LisaLlama3MetaModel.__init__(self, config, **kwargs)


class LisaLlama3ForCausalLM(MllamaForConditionalGeneration):
    """
    Llama 3.2 Vision (Mllama)ベースのLISA因果言語モデル
    """
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        
        # SAM統合のためのメタモデル初期化
        self.lisa_meta = LisaLlama3MetaModel(config, **kwargs)
        
        # SAMモデルの参照をメインモデルにも設定
        self.sam = self.lisa_meta.sam
        self.seg_projection = self.lisa_meta.seg_projection
        self.seg_token_idx = self.lisa_meta.seg_token_idx
        self.dice_loss_weight = self.lisa_meta.dice_loss_weight
        self.bce_loss_weight = self.lisa_meta.bce_loss_weight
        
        # Processor (AutoProcessor) - 初期化だけしておき、後でload_processで設定
        self.processor = None
        
    def load_processor(self, model_name_or_path):
        """
        Llama 3.2 Vision用のプロセッサをロード
        """
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        # トークナイザーの初期サイズを記録
        initial_vocab_size = len(self.processor.tokenizer)
        print(f"初期トークナイザーサイズ: {initial_vocab_size}")
        
        # セグメントトークンをプロセッサに追加
        # add_tokensを使用
        num_added_tokens = self.processor.tokenizer.add_tokens([SEG_TOKEN])
        print(f"追加されたトークン数: {num_added_tokens}")
        
        # トークナイザーが更新されたことを確認
        new_vocab_size = len(self.processor.tokenizer)
        print(f"新しいトークナイザーサイズ: {new_vocab_size}")
        
        try:
            # 語彙サイズの調整（新トークン追加に伴う）
            print("トークン埋め込みをリサイズします...")
            self.resize_token_embeddings(new_vocab_size)
            print("トークン埋め込みのリサイズが完了しました")
        except Exception as e:
            print(f"トークン埋め込みのリサイズ中にエラーが発生しました: {e}")
            raise
        
        # SEGトークンのインデックスを保存
        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        print(f"SEGトークンのインデックス: {self.seg_token_idx}")
        
    def prepare_inputs_for_sam(self, image):
        """
        SAMモデル用の入力を準備
        """
        # SAMの入力形式に変換
        sam_image = self.sam.preprocess(image)
        return sam_image
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        decoder_input_ids=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        モデルの順伝播計算
        """
        # 基本の前方伝播処理
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            past_key_values=past_key_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 学習時かつセグメンテーション学習用マスクが提供されている場合、セグメンテーション損失を計算
        if labels is not None and "seg_masks" in kwargs and self.seg_token_idx is not None:
            # 最後の隠れ状態を取得
            hidden_states = outputs.hidden_states[-1]
            
            # <seg>トークンの位置を検出
            seg_positions = (input_ids == self.seg_token_idx).nonzero(as_tuple=True)
            
            # <seg>トークン位置の隠れ状態を抽出
            if seg_positions[0].shape[0] > 0:
                batch_indices, seq_indices = seg_positions
                
                # SAMへの射影ベクトルを計算
                sam_prompt_embeddings = self.seg_projection(
                    hidden_states[batch_indices, seq_indices]
                )
                
                # batch_indices（バッチ内の位置）に基づいて、対応する画像と正解マスクを取得
                # この例では単純化のため、1つの<seg>トークンに1つのマスクが対応すると仮定
                loss = self._compute_segmentation_loss(
                    sam_prompt_embeddings,
                    pixel_values,  # 元の画像
                    kwargs["seg_masks"],  # 正解マスク
                    batch_indices  # どのバッチ要素に対応するか
                )
                
                # 言語モデルの損失とセグメンテーション損失を組み合わせる
                outputs.loss = outputs.loss + loss
                
        return outputs
    
    def _compute_segmentation_loss(
        self, 
        prompt_embeddings, 
        images, 
        target_masks, 
        batch_indices
    ):
        """
        SAMを使用してセグメンテーション損失を計算
        
        Args:
            prompt_embeddings: <seg>トークンから生成された埋め込みベクトル
            images: 入力画像
            target_masks: 正解セグメンテーションマスク
            batch_indices: バッチ内の位置インデックス
        """
        # SAM用の画像特徴量を計算
        image_embeddings = self.sam.image_encoder(images)
        
        # 各プロンプト埋め込みについて損失を計算
        total_loss = 0
        total_masks = 0
        
        for i, embedding in enumerate(prompt_embeddings):
            batch_idx = batch_indices[i].item()
            
            # SAMにプロンプト埋め込みを入力し、マスク予測を取得
            mask_predictions, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[batch_idx].unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=embedding.unsqueeze(0),
                dense_prompt_embeddings=None,
                multimask_output=False,
            )
            
            # 対応する正解マスク
            target_mask = target_masks[batch_idx]
            
            # マスク損失計算（Dice損失とBCE損失の組み合わせ）
            dice_loss = self._dice_loss(
                mask_predictions, 
                target_mask.unsqueeze(0).unsqueeze(0),
                num_masks=1
            )
            bce_loss = self._sigmoid_ce_loss(
                mask_predictions, 
                target_mask.unsqueeze(0).unsqueeze(0),
                num_masks=1
            )
            
            mask_loss = (
                self.dice_loss_weight * dice_loss + 
                self.bce_loss_weight * bce_loss
            )
            
            total_loss += mask_loss
            total_masks += 1
            
        # マスクの平均損失を返す
        if total_masks > 0:
            return total_loss / total_masks
        else:
            # <seg>トークンが出現しない場合は損失なし
            return 0.0
    
    def _dice_loss(self, inputs, targets, num_masks, scale=1000, eps=1e-6):
        """
        Dice損失の計算
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1, 2)
        targets = targets.flatten(1, 2)
        numerator = 2 * (inputs / scale * targets).sum(-1)
        denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
        loss = 1 - (numerator + eps) / (denominator + eps)
        loss = loss.sum() / (num_masks + 1e-8)
        return loss
    
    def _sigmoid_ce_loss(self, inputs, targets, num_masks):
        """
        シグモイドBCE損失の計算
        """
        inputs = inputs.flatten(1, 2)
        targets = targets.flatten(1, 2)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = loss.mean(1).sum() / (num_masks + 1e-8)
        return loss
    
    def generate_segmentation(
        self,
        image,
        prompt,
        max_new_tokens=32,
        **generate_kwargs
    ):
        """
        テキストプロンプトに基づいてセグメンテーションマスクを生成
        
        Args:
            image: 入力画像
            prompt: テキストプロンプト
            max_new_tokens: 生成する最大トークン数
        """
        if self.processor is None:
            raise ValueError("プロセッサがロードされていません。先にload_processor()を呼び出してください。")
        
        # チャットメッセージフォーマットの作成
        messages = [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
        
        # メッセージをチャットテンプレートに変換
        input_text = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            return_tensors=None
        )
        
        # 画像とテキストの処理
        model_inputs = self.processor(
            image, 
            input_text, 
            return_tensors="pt"
        )
        
        # デバイス移動
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(self.device)
        
        # SAM用の画像埋め込みを計算
        sam_image = self.prepare_inputs_for_sam(image)
        sam_image_embedding = self.sam.image_encoder(sam_image.to(self.device))
        
        # トークン生成
        outputs = self.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **generate_kwargs
        )
        
        # 生成されたトークン
        generated_ids = outputs.sequences[0]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=False)
        
        # <seg>トークンの検出
        seg_positions = (generated_ids == self.seg_token_idx).nonzero(as_tuple=True)[0]
        
        # <seg>トークンが検出されなかった場合
        if len(seg_positions) == 0:
            return {
                "generated_text": generated_text,
                "segmentation_masks": None
            }
        
        # 隠れ状態から<seg>トークンの位置の埋め込みを抽出
        # hidden_statesの構造に注意 - 生成の場合には各タイムステップごとの隠れ状態がある
        masks = []
        
        for pos in seg_positions:
            # 該当ステップの隠れ状態
            hidden_state = outputs.hidden_states[-1][pos]
            
            # SAM用の射影
            prompt_embedding = self.seg_projection(hidden_state)
            
            # マスク生成
            mask_predictions, _ = self.sam.mask_decoder(
                image_embeddings=sam_image_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=prompt_embedding.unsqueeze(0),
                dense_prompt_embeddings=None,
                multimask_output=False,
            )
            
            # シグモイド適用してマスク取得
            mask = torch.sigmoid(mask_predictions[0, 0])
            masks.append(mask.cpu().detach())
        
        return {
            "generated_text": generated_text,
            "segmentation_masks": masks
        } 