#!/bin/bash

# DeepSpeedを使って8台のGPUでデモを実行するスクリプト

# 使用可能なGPUの数を取得
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "利用可能なGPU数: ${NUM_GPUS}"

# 引数のチェック
if [ "$#" -lt 2 ]; then
    echo "使用方法: $0 <image_path> <prompt> [追加オプション]"
    echo "例: $0 sample.jpg \"Segment all objects in this image\""
    exit 1
fi

IMAGE_PATH=$1
PROMPT=$2
shift 2

# モデルパス
MODEL_PATH="meta-llama/Llama-3.2-11B-Vision-Instruct"

# SAMチェックポイント（設定されている場合）
SAM_CHECKPOINT=""
if [ -f "sam_vit_h_4b8939.pth" ]; then
    SAM_CHECKPOINT="sam_vit_h_4b8939.pth"
    echo "SAMチェックポイントを使用: $SAM_CHECKPOINT"
fi

# 出力ディレクトリ
OUTPUT_DIR="outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "出力ディレクトリ: $OUTPUT_DIR"
echo "プロンプト: $PROMPT"

# 追加オプションの準備
EXTRA_OPTS=""
for opt in "$@"; do
    EXTRA_OPTS="$EXTRA_OPTS $opt"
done

# GPUの数に応じてDeepSpeedのnprocを設定（最大8）
NPROC=$(( NUM_GPUS > 8 ? 8 : NUM_GPUS ))

echo "DeepSpeedを使用して${NPROC}台のGPUで実行します..."

# メインのコマンド実行
# nproc_per_nodeでGPUの数を指定し、deepspeedコマンドでマルチGPU実行
deepspeed --num_gpus=$NPROC demo.py \
    --model_path $MODEL_PATH \
    --sam_checkpoint $SAM_CHECKPOINT \
    --image_path $IMAGE_PATH \
    --prompt "$PROMPT" \
    --output_dir $OUTPUT_DIR \
    --use_deepspeed \
    --ds_config ds_config.json \
    --max_new_tokens 512 \
    --beam_size 1 \
    --temp 0.2 \
    --top_p 0.7 \
    $EXTRA_OPTS

echo "処理が完了しました。結果は $OUTPUT_DIR に保存されています。" 