import argparse
import os
import time
import json
from pathlib import Path
from datetime import datetime
import traceback

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from transformers import AutoProcessor
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
    
    # DeepSpeed 関連のパラメータを追加
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for multi-GPU inference"
    )
    parser.add_argument(
        "--ds_config",
        type=str,
        default="ds_config.json",
        help="Path to DeepSpeed configuration file"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    return parser.parse_args()


def load_model(args):
    """
    Load LISA model and processor
    """
    print(f"Loading model from {args.model_path}...")
    
    # トーチの精度を下げてメモリ使用量を減らす
    torch.set_float32_matmul_precision('medium')
    
    # GPUメモリをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU使用可能: {torch.cuda.get_device_name(0)}")
        print(f"現在のGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    try:
        # DeepSpeedを使用する場合
        if args.use_deepspeed:
            print("DeepSpeedを使用してマルチGPUモードで実行します")
            
            # DeepSpeedの設定をロード
            with open(args.ds_config, "r") as f:
                ds_config = json.load(f)
            
            # ローカルランクを設定
            local_rank = args.local_rank
            if local_rank == -1:
                # deepspeed起動コマンドからローカルランクを取得
                local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
                
            if local_rank != -1:
                print(f"ローカルランク: {local_rank}")
                # GPUを初期化
                torch.cuda.set_device(local_rank)
            
            # オートスケール設定を有効にする
            ds_config["zero_optimization"]["stage"] = 3
            ds_config["train_micro_batch_size_per_gpu"] = 1
            
            # モデルをロード
            model = LISAForCausalLM.from_pretrained(
                args.model_path,
                sam_checkpoint=args.sam_checkpoint,
                seg_token=args.seg_token,
                use_deepspeed=True,
                ds_config=ds_config,
                local_rank=local_rank,
                max_batch_size=1
            )
            
            # DeepSpeedモデルの情報を表示
            if hasattr(model, "model") and hasattr(model.model, "module"):
                print(f"DeepSpeedエンジンが初期化されました: {type(model.model).__name__}")
        else:
            # 通常のモード（単一GPU）
            model = LISAForCausalLM.from_pretrained(
                args.model_path,
                sam_checkpoint=args.sam_checkpoint,
                seg_token=args.seg_token,
                device=args.device,
                max_batch_size=1  # バッチサイズを1に制限
            )
        
        return model
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {str(e)}")
        import traceback
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
    """Visualize segmentation results"""
    print(f"Saving visualization to {output_path}...")
    
    # Create figure
    if len(masks) == 0:
        # マスクがない場合は元の画像だけ表示
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title("Original Image (No Segmentation Masks)")
        plt.axis("off")
    else:
        # マスクがある場合は元の画像とマスクを表示
        fig, axs = plt.subplots(1 + len(masks), 1, figsize=(10, 10 + 5 * len(masks)))
        
        # Original image
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        
        # Segmentation masks
        for i, mask in enumerate(masks):
            # Convert tensor to numpy
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # もしマスクが辞書の場合は、segmentationフィールドを使用
            if isinstance(mask, dict) and "segmentation" in mask:
                mask = mask["segmentation"]
            
            # マスクの形状を表示
            print(f"処理中のマスク形状: {mask.shape}, 型: {type(mask)}")
            
            # 画像とマスクのサイズが一致しない場合はリサイズ
            image_array = np.array(image)
            img_h, img_w = image_array.shape[:2]
            mask_h, mask_w = mask.shape[:2]
            
            if mask_h != img_h or mask_w != img_w:
                print(f"マスクサイズ ({mask_h}x{mask_w}) を画像サイズ ({img_h}x{img_w}) に合わせてリサイズします")
                import cv2
                mask = cv2.resize(mask.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.float32)  # 閾値を適用して2値化
            
            # 次元数が一致していない場合の対応（3次元マスクを2次元に変換）
            if len(mask.shape) > 2:
                if mask.shape[0] == 1:  # バッチ次元が含まれている場合
                    mask = mask[0]  # 最初の次元を削除
                if len(mask.shape) > 2:  # まだ2次元以上の場合
                    print(f"マスクの次元が多すぎます: {mask.shape}、次元を圧縮します")
                    # 値が存在する場所をすべて1に設定（任意の次元から2次元マスクを作成）
                    mask = (mask.sum(axis=tuple(range(len(mask.shape)-2))) > 0).astype(np.float32)
            
            # Create a blended visualization
            vis_image = np.array(image).copy()
            mask_colored = np.zeros_like(vis_image, dtype=np.uint8)
            
            # Create colored mask
            color = np.random.randint(0, 255, (3,))
            for c in range(3):
                mask_colored[:, :, c] = mask * color[c]
            
            # Blend mask with image
            alpha = 0.5
            vis_image = (1 - alpha) * vis_image + alpha * mask_colored
            vis_image = vis_image.astype(np.uint8)
            
            # Display blended image
            axs[i+1].imshow(vis_image)
            axs[i+1].set_title(f"Segmentation Mask {i+1}")
            axs[i+1].axis("off")
    
    # Add text below the visualization if available
    if text:
        plt.figtext(0.5, 0.01, text, ha="center", fontsize=12, wrap=True)
    
    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main(args):
    """
    Main function
    """
    # ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # モデルのロード
        model = load_model(args)
        
        # 画像のロード
        print(f"Loading image from {args.image_path}...")
        image = load_image(args.image_path, resize_factor=args.resize_factor)
        
        # 推論実行
        print(f"Running inference with prompt: '{args.prompt}'...")
        
        # プロンプト
        prompt = args.prompt
        
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
            
            # セグメンテーション生成
            result = model.generate_segmentation(
                image=image,
                prompt=prompt,
                **generation_params
            )
            
            # 結果を表示
            if isinstance(result, str):
                # 文字列の場合はそのまま表示
                print(f"生成されたテキスト: {result}")
                output_path = os.path.join(args.output_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"結果を保存しました: {output_path}")
            elif isinstance(result, dict) and 'masks' in result:
                # マスクと文字列を含む辞書の場合
                masks = result.get('masks', [])
                text = result.get('text', '')
                
                # テキスト内容をデバッグ表示
                print(f"生成テキストの長さ: {len(text)} 文字")
                print(f"生成テキストの先頭部分: {text[:100]}...")
                
                # テキストを別ファイルに保存
                if text:
                    text_output_path = os.path.join(args.output_dir, f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    with open(text_output_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"生成テキストを保存しました: {text_output_path}")
                
                # 結果の可視化と保存
                output_path = os.path.join(args.output_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                visualize_results(image, masks, text, output_path)
                print(f"結果を保存しました: {output_path}")
            else:
                print(f"未知の結果形式: {type(result)}")
                
        except Exception as e:
            print(f"推論中にエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # deepspeedコマンドで実行される場合、自動的にローカルランクが設定される
    # 通常のpythonコマンドで実行される場合は、DeepSpeedを使用しない
    args = parse_args()
    
    # DeepSpeedで初期化
    if args.use_deepspeed:
        deepspeed.init_distributed()
    
    main(args) 