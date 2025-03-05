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
from transformers import AutoProcessor

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
        # Load LISA model
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
            
            # Display
            axs[i + 1].imshow(vis_image)
            axs[i + 1].set_title(f"Segmentation Mask {i+1}")
            axs[i + 1].axis("off")
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Save text output
    text_path = output_path.replace(".png", ".txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)


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
                'temperature': 0.2,
                'top_p': 0.7,
                'repetition_penalty': 1.2,
                'num_beams': 1,
                'max_new_tokens': min(args.max_new_tokens, 100)  # 最大100トークンに制限
            }
            
            # セグメンテーション生成
            result = model.generate_segmentation(
                image=image,
                prompt=prompt,
                **generation_params
            )
            
            # 結果を表示
            if 'masks' in result and result['masks']:
                print(f"セグメンテーションマスクが生成されました: {len(result['masks'])}個")
                
                # 結果を保存
                os.makedirs(args.output_dir, exist_ok=True)
                
                # 元の画像をコピー
                image_pil = Image.fromarray(np.array(image))
                image_pil.save(f"{args.output_dir}/original_image.png")
                
                # マスク画像を保存
                for i, mask in enumerate(result['masks']):
                    mask_image = np.zeros_like(np.array(image))
                    mask_image[mask] = np.array(image)[mask]
                    
                    # マスク画像を保存
                    mask_pil = Image.fromarray(mask_image)
                    mask_pil.save(f"{args.output_dir}/mask_{i}.png")
                    
                    # 可視化用の色付きマスク
                    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    colored_mask[mask] = [255, 0, 0]  # 赤色でマスク
                    
                    # 元画像に重ねる
                    overlay = np.array(image).copy()
                    overlay = overlay * 0.7 + colored_mask * 0.3
                    
                    # 保存
                    overlay_pil = Image.fromarray(overlay.astype(np.uint8))
                    overlay_pil.save(f"{args.output_dir}/overlay_{i}.png")
                
            # テキスト出力
            if 'text' in result:
                print(f"生成されたテキスト: {result['text']}")
                
                # テキストを保存
                with open(f"{args.output_dir}/output.txt", "w", encoding="utf-8") as f:
                    f.write(result['text'])
            else:
                print("テキスト出力がありません")
        
        except Exception as e:
            print(f"セグメンテーション生成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # GPUメモリの解放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPUメモリ使用状況（推論後）: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


if __name__ == "__main__":
    main(parse_args()) 