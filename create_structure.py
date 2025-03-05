#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ディレクトリ構造の定義
directories = [
    "lisa",
    "lisa/models",
    "lisa/models/mllama",
    "lisa/models/sam",
    "lisa/utils",
    "lisa/data",
    "lisa/train",
    "lisa/eval",
    "weights",
    "outputs",
    "logs",
    "configs",
]

# 基本ファイルのリスト
base_files = [
    "lisa/__init__.py",
    "lisa/models/__init__.py",
    "lisa/models/mllama/__init__.py",
    "lisa/models/sam/__init__.py",
    "lisa/utils/__init__.py",
    "lisa/data/__init__.py",
    "lisa/train/__init__.py",
    "lisa/eval/__init__.py",
]

# ディレクトリ作成
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"ディレクトリ作成: {directory}")

# 基本ファイル作成
for file_path in base_files:
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("# -*- coding: utf-8 -*-\n")
        print(f"ファイル作成: {file_path}")

print("プロジェクト構造の作成が完了しました。") 