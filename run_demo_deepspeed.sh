#!/bin/bash

# DeepSpeedを使って複数GPUでデモを実行するスクリプト

# DeepSpeedログレベル設定（1=INFO, 2=DEBUG, 3=WARNING, 4=ERROR, 5=CRITICAL）
# エラーメッセージと埋め込み拡張関連のログ表示を確認するため、INFOレベルに設定
export DEEPSPEED_LOGGER_LEVEL=1
export DS_ACCELERATOR=cuda       # 明示的にアクセラレータを設定

# CUDAカーネルエラーの詳細なトレース情報を取得するための設定
# CUDA操作を同期的に実行し、エラー発生時に正確な位置を特定可能にする
export CUDA_LAUNCH_BLOCKING=1

# 使用可能なGPUの数を取得
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "利用可能なGPU数: ${NUM_GPUS}"

# 引数のチェック
if [ "$#" -lt 2 ]; then
    echo "使用方法: $0 <image_path> <prompt> [追加オプション]"
    echo "追加オプション:"
    echo "  --single-gpu    単一GPUモードで実行（メモリ効率重視）"
    echo "例: $0 sample.jpg \"Segment all objects in this image\""
    echo "例: $0 sample.jpg \"Segment all objects in this image\" --single-gpu"
    exit 1
fi

IMAGE_PATH=$1
PROMPT=$2
shift 2

# 単一GPUモードのチェック
SINGLE_GPU=0
EXTRA_OPTS=""
for opt in "$@"; do
    if [ "$opt" == "--single-gpu" ]; then
        SINGLE_GPU=1
    else
        EXTRA_OPTS="$EXTRA_OPTS $opt"
    fi
done

# モデルパス
MODEL_PATH="meta-llama/Llama-3.2-11B-Vision-Instruct"

# SAMチェックポイント（設定されている場合）
SAM_CHECKPOINT=""
# 複数の場所をチェック
if [ -f "sam_vit_h_4b8939.pth" ]; then
    SAM_CHECKPOINT="sam_vit_h_4b8939.pth"
elif [ -f "checkpoints/sam_vit_h_4b8939.pth" ]; then
    SAM_CHECKPOINT="checkpoints/sam_vit_h_4b8939.pth"
fi

if [ ! -z "$SAM_CHECKPOINT" ]; then
    echo "SAMチェックポイントを使用: $SAM_CHECKPOINT"
fi

# 出力ディレクトリ
OUTPUT_DIR="outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# ログファイルの設定
LOG_FILE="${OUTPUT_DIR}/deepspeed_log.txt"
echo "ログは ${LOG_FILE} に保存されます"

echo "出力ディレクトリ: $OUTPUT_DIR"
echo "プロンプト: $PROMPT"

# GPUの数に応じてDeepSpeedのnprocを設定
if [ $SINGLE_GPU -eq 1 ]; then
    echo "単一GPUモードで実行します..."
    NPROC=1
else
    # 最大8台のGPUを使用
    NPROC=$(( NUM_GPUS > 8 ? 8 : NUM_GPUS ))
    if [ $NPROC -lt 1 ]; then
        NPROC=1  # 少なくとも1つのGPUが必要
    fi
    echo "DeepSpeedを使用して${NPROC}台のGPUで実行します..."
fi

# SAMチェックポイントオプションの準備
SAM_OPTS=""
if [ ! -z "$SAM_CHECKPOINT" ]; then
    SAM_OPTS="--sam_checkpoint $SAM_CHECKPOINT"
fi

# DS_CONFIGの存在確認
if [ ! -f "ds_config.json" ]; then
    echo "エラー: ds_config.json が見つかりません"
    exit 1
fi

# メインのコマンド実行
echo "モデルを読み込み中..."

# メモリ不足対策のための環境変数設定
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# マスターポートをランダムに設定して競合を避ける
MASTER_PORT=$(( 10000 + RANDOM % 50000 ))

# nproc_per_nodeでGPUの数を指定し、deepspeedコマンドでマルチGPU実行
deepspeed --num_gpus=$NPROC --no_local_rank --master_port $MASTER_PORT demo.py \
    --model_path $MODEL_PATH \
    $SAM_OPTS \
    --image_path $IMAGE_PATH \
    --prompt "$PROMPT" \
    --output_dir $OUTPUT_DIR \
    --use_deepspeed \
    --ds_config ds_config.json \
    --max_new_tokens 512 \
    --beam_size 1 \
    --temp 0.2 \
    --top_p 0.7 \
    $EXTRA_OPTS 2>&1 | tee $LOG_FILE

# 実行結果のチェック
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo "処理が完了しました。結果は $OUTPUT_DIR に保存されています。"
else
    echo "エラーが発生しました（コード: $EXIT_CODE）。ログファイル $LOG_FILE を確認してください。"
fi 