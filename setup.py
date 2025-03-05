#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse

def run_command(command):
    # コマンド実行関数
    print(f"実行コマンド: {command}")
    subprocess.run(command, shell=True, check=True)

def setup_environment():
    # Conda環境のセットアップ
    print("Conda環境をセットアップしています...")
    run_command("conda env create -f environment.yaml")
    print("環境セットアップ完了！")
    
    # Hugging Faceログイン
    print("Hugging Faceにログインしています...")
    run_command("huggingface-cli login --token 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")
    
    # 必要なディレクトリの作成
    dirs = ["weights", "outputs", "logs"]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"ディレクトリ作成: {dir_name}")

def main():
    parser = argparse.ArgumentParser(description="LISA-Llama環境セットアップツール")
    parser.add_argument("--skip-env", action="store_true", help="環境構築をスキップする")
    
    args = parser.parse_args()
    
    if not args.skip_env:
        setup_environment()
    
    print("セットアップが完了しました！")
    print("環境を有効化するには: conda activate lisa-llama")

if __name__ == "__main__":
    main() 