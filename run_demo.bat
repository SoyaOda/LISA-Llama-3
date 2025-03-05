@echo off
chcp 932
python demo.py ^
    --image_path ./imgs/example.jpg ^
    --model_path meta-llama/Llama-3.2-11B-Vision-Instruct ^
    --sam_checkpoint C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth ^
    --prompt "この画像にある食べ物を全てセグメンテーションして、それぞれの栄養価を説明してください。<seg>"

echo.
echo デモの実行が完了しました。
echo 出力は ./outputs ディレクトリに保存されました。
pause 