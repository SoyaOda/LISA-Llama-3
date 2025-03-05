@echo off
chcp 932
python train_ds.py ^
    --version meta-llama/Llama-3.2-11B-Vision-Instruct ^
    --sam_checkpoint C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth ^
    --batch_size 1 ^
    --grad_accumulation_steps 4 ^
    --lora_r 8 ^
    --lora_alpha 16 ^
    --epochs 5 ^
    --steps_per_epoch 100

echo.
echo トレーニングが完了しました。
echo モデルチェックポイントは ./checkpoints ディレクトリに保存されました。
pause 