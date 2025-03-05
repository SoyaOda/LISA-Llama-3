import argparse
import os
import shutil
import sys
import time
from functools import partial

import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor

from model.LISA import LISAForCausalLM
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="meta-llama/Llama-3.2-11B-Vision-Instruct"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--sam_checkpoint", default="C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--seg_token_idx", default=32007, type=int, help="segmentation token index"
    )
    parser.add_argument("--use_deepspeed", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed_config",
        default="./scripts/zero3.json",
        type=str,
        help="deepspeed config",
    )
    parser.add_argument(
        "--unfreeze_llm", action="store_true", default=False, help="unfreeze LLM backbone"
    )

    return parser.parse_args(args)


def main(args):
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        if not os.path.exists(args.vis_save_path):
            os.makedirs(args.vis_save_path, exist_ok=True)

    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = torch.float32

    quantization_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    model_args = {
        "torch_dtype": dtype,
        "quantization_config": quantization_config,
        "sam_checkpoint": args.sam_checkpoint, 
        "seg_token_idx": args.seg_token_idx,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
    }
    
    model = LISAForCausalLM.from_pretrained(
        args.version,
        **model_args
    )
    model.load_processor(args.version)

    def find_linear_layers(model, lora_target_modules):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls) and any(
                target_key in name for target_key in lora_target_modules
            ):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    lora_target_modules = args.lora_target_modules.split(",")
    if args.lora_target_modules:
        lora_module_names = find_linear_layers(model, lora_target_modules)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_module_names,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = HybridDataset(
        args=args,
        tokenizer=model.processor.tokenizer,
        vision_tower=None,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch,
        is_train=True,
    )

    if not args.exclude_val:
        val_dataset = ValDataset(
            args=args,
            tokenizer=model.processor.tokenizer,
            vision_tower=None,
        )
    else:
        val_dataset = None

    sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=partial(collate_fn, tokenizer=model.processor.tokenizer),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    
    if val_dataset is not None:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            sampler=val_sampler,
            collate_fn=partial(collate_fn, tokenizer=model.processor.tokenizer),
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        val_loader = None
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    
    device = torch.device("cuda", args.local_rank)
    model = model.to(device)
    
    writer = None
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)
    
    train_iter = 0
    start_epoch = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location="cpu")
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            train_iter = checkpoint["train_iter"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    
    if args.auto_resume:
        resume_file = os.path.join(args.log_dir, "checkpoint.pth")
        if os.path.exists(resume_file):
            print(f"=> loading checkpoint '{resume_file}'")
            checkpoint = torch.load(resume_file, map_location="cpu")
            start_epoch = checkpoint["epoch"] 
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            train_iter = checkpoint["train_iter"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume_file, checkpoint["epoch"]
                )
            )
    
    if not args.eval_only:
        for epoch in range(start_epoch, args.epochs):
            train_iter = train(
                train_loader,
                model,
                epoch,
                scheduler,
                writer,
                train_iter,
                args,
                optimizer,
            )
            
            if not args.no_eval and val_loader is not None:
                validate(val_loader, model, epoch, writer, args)
            
            if args.local_rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "train_iter": train_iter,
                    },
                    args.log_dir,
                )
    else:
        if val_loader is not None:
            validate(val_loader, model, 0, writer, args)


def save_checkpoint(state, log_dir, filename="checkpoint.pth"):
    filename = os.path.join(log_dir, filename)
    torch.save(state, filename)
    print(f"Saved checkpoint to {filename}")


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
    optimizer,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time,
            data_time,
            losses,
            ce_losses,
            mask_bce_losses,
            mask_dice_losses,
            mask_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    
    num_steps = len(train_loader)
    end = time.time()
    
    for i, (images, images_clip, input_ids, labels, attention_masks, offset, masks_list, resize_list) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        cur_batch_size = images.size(0)
        
        dict_to_cuda({
            "images": images,
            "images_clip": images_clip,
            "input_ids": input_ids, 
            "labels": labels,
            "attention_masks": attention_masks,
            "masks_list": masks_list
        })
        
        loss, logits = model.model_forward(
            images=images,
            images_clip=images_clip,
            input_ids=input_ids,
            labels=labels,
            attention_masks=attention_masks,
            offset=offset,
            masks_list=masks_list,
            label_list=masks_list,
            resize_list=resize_list,
            inference=False,
        )
        
        loss = loss / args.grad_accumulation_steps
        
        loss.backward()
        
        if (i + 1) % args.grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        losses.update(loss.item(), cur_batch_size)
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)
            if writer is not None:
                writer.add_scalar("train/loss", losses.avg, train_iter)
                writer.add_scalar(
                    "metrics/total_steps_per_sec", args.batch_size / batch_time.avg, train_iter
                )
                writer.add_scalar(
                    "metrics/steps_per_sec", 1 / batch_time.avg, train_iter
                )
                writer.add_scalar(
                    "metrics/data_time_per_step", data_time.avg, train_iter
                )
                writer.add_scalar("progress/epoch", epoch, train_iter)
                writer.add_scalar("progress/step", i, train_iter)
                writer.add_scalar("progress/total_step", train_iter, train_iter)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], train_iter
                )
        
        train_iter += 1
    
    return train_iter


def validate(val_loader, model, epoch, writer, args):
    batch_time = AverageMeter("Time", ":6.3f")
    stats = {}

    model.eval()
    
    intersection_meter = AverageMeter("Intersec", ":6.3f")
    union_meter = AverageMeter("Union", ":6.3f")
    acc_iou_meter = AverageMeter("gIoU", ":6.3f")

    with torch.no_grad():
        end = time.time()
        for i, (
            images,
            images_clip,
            input_ids,
            attention_masks,
            offset,
            masks_list,
            resize_list,
            original_size_list,
            label_list,
        ) in enumerate(val_loader):
            dict_to_cuda({
                "images": images,
                "images_clip": images_clip,
                "input_ids": input_ids,
                "attention_masks": attention_masks,
                "masks_list": masks_list,
                "label_list": label_list,
            })
            
            text, output_masks = model.evaluate(
                images_clip=images_clip,
                images=images,
                input_ids=input_ids,
                resize_list=resize_list,
                original_size_list=original_size_list,
                max_new_tokens=128,
                tokenizer=model.processor.tokenizer,
            )
            
            for b, (mask_b, gt_mask_b) in enumerate(zip(output_masks, label_list)):
                if len(mask_b) == 0:
                    continue
                
                mask_pred = torch.stack(mask_b).unsqueeze(1)
                mask_gt = gt_mask_b.unsqueeze(1)
                
                intersection, union, _ = intersectionAndUnionGPU(
                    mask_pred, mask_gt, 2
                )
                
                intersection_meter.update(intersection.cpu().numpy())
                union_meter.update(union.cpu().numpy())
                
                iou = intersection.float() / (union.float() + 1e-10)
                acc_iou_meter.update(iou[1].cpu().numpy())
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print(f"Test: [{i}/{len(val_loader)}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})")
    
    IoU = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(IoU)
    
    print(f"* mIoU {mIoU:.3f} gIoU {acc_iou_meter.avg:.3f}")
    
    if writer is not None:
        writer.add_scalar("val/mIoU", mIoU, epoch)
        writer.add_scalar("val/gIoU", acc_iou_meter.avg, epoch)
    
    return stats


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
