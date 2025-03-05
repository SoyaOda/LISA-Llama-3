#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import time
from pathlib import Path
from datetime import datetime
import traceback

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from transformers import AutoProcessor, BitsAndBytesConfig
import deepspeed

from model.LISA import LISAForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Run LISA demo with an image")
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Path to the model"
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--seg_token",
        type=str,
        default="<seg>",
        help="Segmentation token to use"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--resize_factor",
        type=float,
        default=0.8,
        help="Factor to resize the image (e.g., 0.5 for half size, helps reduce memory usage)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for the model"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    # デバッグ用のパラメータを追加
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout for generation in seconds"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size for generation"
    )
    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.7)
    # DeepSpeed関連のオプション
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "bf16", "fp16"],
        help="精度設定（fp32, bf16, fp16）"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=False,
        help="8ビットで量子化してロード"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="4ビットで量子化してロード"
    )
    return parser.parse_args()


def load_model(args):
    """
    Load LISA model and processor with DeepSpeed optimization
    """
    print(f"モデルをロード中: {args.model_path}...")
    
    # トーチの精度を下げてメモリ使用量を減らす
    torch.set_float32_matmul_precision('medium')
    
    # GPUメモリをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU使用可能: {torch.cuda.get_device_name(0)}")
        print(f"現在のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    try:
        # 精度設定
        torch_dtype = torch.float32
        if args.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif args.precision == "fp16":
            torch_dtype = torch.float16
        
        # 量子化設定
        quantization_config = None
        if args.load_in_4bit:
            print("4ビット量子化を使用します")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif args.load_in_8bit:
            print("8ビット量子化を使用します")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # モデルのロード設定
        model_args = {
            "sam_checkpoint": args.sam_checkpoint,
            "seg_token": args.seg_token,
            "device": "meta" if args.precision in ["fp16", "bf16"] else args.device,  # DeepSpeedの場合はmetaデバイスで初期化
            "max_batch_size": 1
        }
        
        # モデルをロード
        model = LISAForCausalLM.from_pretrained(
            args.model_path,
            **model_args
        )
        
        # DeepSpeedによる最適化（FP16/BF16の場合のみ）
        if args.precision in ["fp16", "bf16"] and not args.load_in_4bit and not args.load_in_8bit:
            print(f"DeepSpeed推論最適化を適用中（{args.precision}）...")
            
            # Vision Towerを保存
            vision_tower = None
            if hasattr(model, "get_model") and hasattr(model.get_model(), "get_vision_tower"):
                vision_tower = model.get_model().get_vision_tower()
                model.model.vision_tower = None
            
            # SAMモジュールを保存
            sam_model = None
            if hasattr(model, "sam_model"):
                sam_model = model.sam_model
                model.sam_model = None
            
            # DeepSpeed初期化
            model_engine = deepspeed.init_inference(
                model=model,
                dtype=torch_dtype,
                replace_with_kernel_inject=True,
                replace_method="auto",
            )
            model = model_engine.module
            
            # 保存したモジュールを戻す
            if vision_tower is not None:
                model.model.vision_tower = vision_tower.to(dtype=torch_dtype).to(args.device)
            if sam_model is not None:
                model.sam_model = sam_model.to(args.device)
        else:
            # 通常の方法でデバイスと精度を設定
            model = model.to(dtype=torch_dtype).to(args.device)
        
        print("モデルのロードが完了しました")
        if torch.cuda.is_available():
            print(f"モデルロード後のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        return model
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {str(e)}")
        traceback.print_exc()
        raise


def load_image(image_path, resize_factor=1.0):
    """
    Load image from path
    """
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path)
    
    # 画像をリサイズ（必要な場合）
    if resize_factor != 1.0:
        original_size = image.size
        new_size = (int(original_size[0] * resize_factor), int(original_size[1] * resize_factor))
        print(f"Resizing image from {original_size} to {new_size}")
        image = image.resize(new_size, Image.LANCZOS)
    
    return image


def visualize_results(image, masks, text, output_path):
    """セグメンテーション結果を可視化"""
    print(f"可視化結果を保存: {output_path}...")
    
    # PILイメージをNumPy配列に変換（必要ならば）
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # 行列形状に応じてサブプロットを設定
    n_masks = len(masks)
    fig_cols = min(3, n_masks + 1)  # 最大3列（元画像+マスク）
    fig_rows = max(2, 1 + (n_masks // fig_cols))  # 最低2行（上部に元画像）
    
    fig = plt.figure(figsize=(fig_cols * 5, fig_rows * 5))
    
    # オリジナル画像を表示
    ax = fig.add_subplot(fig_rows, fig_cols, 1)
    ax.imshow(image_np)
    ax.set_title("オリジナル画像", fontsize=14)
    ax.axis('off')
    
    # マスクを表示
    for i, mask in enumerate(masks):
        ax = fig.add_subplot(fig_rows, fig_cols, i + 2)
        
        # 元画像をコピー
        overlay = image_np.copy()
        
        # マスクのあるピクセルにカラーハイライトを適用
        overlay_mask = np.zeros_like(overlay)
        
        # 確率値に応じたマスク（PyTorchテンソルかNumPy配列かを判断）
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
            
        # マスクが確率値の場合は閾値を適用
        if mask_np.dtype != bool:
            mask_np = mask_np > 0.5
        
        # マスク部分を赤色でハイライト（半透明）
        overlay_mask[mask_np] = [255, 0, 0]
        
        # 元画像とマスクを合成
        highlighted = image_np.copy()
        mask_region = mask_np
        highlighted[mask_region] = image_np[mask_region] * 0.5 + overlay_mask[mask_region] * 0.5
        
        # プロット
        ax.imshow(highlighted)
        ax.set_title(f"セグメント {i+1}", fontsize=14)
        ax.axis('off')
    
    # テキスト結果を下部に表示
    text_ax = fig.add_subplot(fig_rows, 1, fig_rows)
    text_ax.text(0.5, 0.5, text, fontsize=12, ha='center', va='center', wrap=True)
    text_ax.axis('off')
    
    # スペースを最適化して保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 各マスクも個別に保存
    output_dir = os.path.dirname(output_path)
    basename = os.path.basename(output_path).split('.')[0]
    
    for i, mask in enumerate(masks):
        # 元画像に個別マスクを適用
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
            
        # マスクが確率値の場合は閾値を適用
        if mask_np.dtype != bool:
            mask_np = mask_np > 0.5
            
        # マスクを適用した画像
        masked_image = image_np.copy()
        masked_image[~mask_np] = masked_image[~mask_np] * 0.3  # マスク外を暗くする
        
        # 保存
        mask_path = os.path.join(output_dir, f"{basename}_mask_{i+1}.png")
        plt.imsave(mask_path, masked_image)


def main(args):
    """
    Main function
    """
    # ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # モデルのロード
        model = load_model(args)
        
        # 画像のロード
        print(f"画像をロード中: {args.image_path}...")
        image = load_image(args.image_path, resize_factor=args.resize_factor)
        
        # 推論実行
        print(f"プロンプト: '{args.prompt}'で推論実行中...")
        
        # プロンプト
        text_prompt = args.prompt
        
        # デバイスを表示
        print(f"デバイス: {args.device}")
        
        # GPUメモリの情報表示
        if torch.cuda.is_available():
            print(f"GPUメモリ使用状況（推論前）: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
        try:
            # 生成パラメータを設定
            generation_params = {
                'do_sample': True,
                'temperature': args.temp,
                'top_p': args.top_p,
                'repetition_penalty': 1.2,
                'num_beams': args.beam_size,
                'max_new_tokens': min(args.max_new_tokens, 100)  # 最大100トークンに制限
            }
            
            # タイムアウト設定（オプション）
            if args.timeout > 0:
                generation_params['timeout_seconds'] = args.timeout
            
            # セグメンテーション生成
            result = model.generate_segmentation(
                image=image,
                text_prompt=text_prompt,
                **generation_params
            )
            
            # 結果を表示
            if 'masks' in result and result['masks']:
                print(f"セグメンテーションマスクが生成されました: {len(result['masks'])}個")
                # 結果の可視化
                output_path = os.path.join(args.output_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                visualize_results(image, result['masks'], result['text'], output_path)
                print(f"結果を保存しました: {output_path}")
                
                # テキスト結果も保存
                text_path = os.path.join(args.output_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                print(f"テキスト結果を保存しました: {text_path}")
            else:
                print("セグメンテーションマスクが生成されませんでした。プロンプトを見直してください。")
                
            # 実行時間を表示
            print(f"実行時間: {time.time() - start_time:.2f}秒")
            
            # 最終GPUメモリ使用量を表示
            if torch.cuda.is_available():
                print(f"最終GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"推論中にエラーが発生しました: {str(e)}")
            traceback.print_exc()
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main(parse_args()) 