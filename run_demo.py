#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Llama 3.2 Vision + SAMモデルを用いたセグメンテーションデモ
CUDA_LAUNCH_BLOCKING=1を使用して同期実行し、詳細なエラー情報を取得
"""

import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from model.LISA import LISAModel

# CUDAエラーを同期的に取得するための設定
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def visualize_segmentation(image_path, masks, save_path=None):
    """セグメンテーション結果を可視化"""
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    
    # マスクを可視化
    if masks:
        for i, mask in enumerate(masks):
            # マスクを表示
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
            colored_mask[mask > 0] = np.array([30, 144, 255, 128]) / 255.0  # 半透明の青
            plt.imshow(colored_mask, alpha=0.5)
            
            # マスクの境界を計算
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) > 0 and len(mask_indices[1]) > 0:
                y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
                x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
                
                # 境界のバウンディングボックスを表示
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                edgecolor='red', facecolor='none', linewidth=2)
                plt.gca().add_patch(rect)
                
                # マスク番号をラベルとして表示
                plt.text(x_min, y_min - 10, f"mask {i+1}", color='white', 
                         bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    
    # 保存または表示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"結果を保存しました: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="LISA（Llama 3.2 Vision + SAM）デモ")
    parser.add_argument("--image", type=str, required=True, help="セグメンテーションする画像パス")
    parser.add_argument("--prompt", type=str, default="この画像の中の人物をセグメンテーションしてください。<seg>", 
                      help="プロンプト文（<seg>トークンを含めること）")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct", 
                      help="Llama 3.2 Visionモデルのパス")
    parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_h_4b8939.pth", 
                      help="SAMモデルのチェックポイントパス")
    parser.add_argument("--output", type=str, default=None, 
                      help="出力画像のパス（指定しない場合は表示のみ）")
    parser.add_argument("--use_simple_generation", action="store_true", 
                      help="シンプルな生成方法を使用する（エラーが出る場合に推奨）")
    parser.add_argument("--use_deepspeed", action="store_true", 
                      help="DeepSpeedを使用する")
    parser.add_argument("--use_half_precision", action="store_true", 
                      help="半精度（float16）を使用する")
    
    args = parser.parse_args()
    
    print(f"画像: {args.image}")
    print(f"プロンプト: {args.prompt}")
    print(f"モデル: {args.model}")
    print(f"SAMチェックポイント: {args.sam_checkpoint}")
    print(f"シンプル生成: {args.use_simple_generation}")
    print(f"DeepSpeed使用: {args.use_deepspeed}")
    print(f"半精度使用: {args.use_half_precision}")
    
    # PyTorchとCUDAのバージョンを表示
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # モデルの初期化
    try:
        print("LISAモデルを初期化しています...")
        model = LISAModel(
            llm_path=args.model,
            sam_checkpoint=args.sam_checkpoint,
            use_deepspeed=args.use_deepspeed,
            seg_token="<seg>",
            device=None,  # 自動検出
            local_rank=0,
            use_simple_generation=args.use_simple_generation,
            dtype=torch.float16 if args.use_half_precision else torch.float32
        )
        print("モデル初期化完了")
        
        # 画像をロード
        image = Image.open(args.image).convert("RGB")
        
        # セグメンテーション実行
        print("セグメンテーション実行中...")
        result = model.generate_segmentation(
            image=image,
            prompt=args.prompt,
            max_new_tokens=512,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            top_p=0.9,
            top_k=30,
            repetition_penalty=1.0
        )
        
        print("\n============= 結果 ===============")
        print(f"生成テキスト: {result['text']}")
        print(f"検出されたマスク数: {len(result['masks'])}")
        
        # 結果の可視化
        visualize_segmentation(args.image, result['masks'], args.output)
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 