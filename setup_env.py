import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Setup environment for LISA-Llama3.2")
    parser.add_argument(
        "--conda", action="store_true", default=False, help="Use conda environment"
    )
    parser.add_argument(
        "--hf_token", 
        default="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",
        help="Hugging Face token for model download"
    )
    return parser.parse_args()


def setup_conda_environment():
    """Set up conda environment from environment.yaml"""
    print("Setting up conda environment from environment.yaml...")
    subprocess.run(
        ["conda", "env", "create", "-f", "environment.yaml"], 
        check=True
    )
    print("Conda environment setup complete.")


def setup_pip_environment():
    """Install requirements using pip"""
    print("Installing requirements with pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True
    )
    print("Pip requirements installation complete.")


def login_huggingface(token):
    """Login to Hugging Face"""
    print("Logging in to Hugging Face...")
    subprocess.run(
        ["huggingface-cli", "login", "--token", token],
        check=True
    )
    print("Hugging Face login successful.")


def main():
    args = parse_args()
    
    # Setup environment
    if args.conda:
        setup_conda_environment()
    else:
        setup_pip_environment()
    
    # Login to Hugging Face
    login_huggingface(args.hf_token)
    
    print("\nSetup complete! Now you can train LISA with Llama 3.2 Vision.")
    print("\nExample usage:")
    print("\n# トレーニング実行例:")
    print("python train_ds.py --version meta-llama/Llama-3.2-11B-Vision-Instruct \\")
    print("    --sam_checkpoint C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth \\")
    print("    --batch_size 1 --grad_accumulation_steps 4 \\")
    print("    --lora_r 8 --lora_alpha 16 \\")
    print("    --epochs 5 --steps_per_epoch 100")
    
    print("\n# デモ実行例:")
    print("python demo.py --image_path ./imgs/example.jpg \\")
    print("    --model_path meta-llama/Llama-3.2-11B-Vision-Instruct \\")
    print("    --sam_checkpoint C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth \\")
    print("    --prompt \"この画像にある食べ物を全てセグメンテーションして、それぞれの栄養価を説明してください。<seg>\" \\")
    print("    --load_in_4bit")


if __name__ == "__main__":
    main() 