#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-Image Text Encoder Cache Script (Multi-GPU with spawn mode)

支持多卡并行处理，使用 spawn 模式避免 CUDA 冲突。

Usage:
    python scripts/cache_text_encoder_standalone.py \
        --text_encoder /path/to/qwen_3_4b \
        --input_dir /path/to/images \
        --output_dir /path/to/cache
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.multiprocessing as mp
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Z-Image architecture identifier
ARCHITECTURE = "zi"


def find_images(input_dir: str) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def get_caption(image_path: Path) -> Optional[str]:
    """获取图片对应的文本描述"""
    txt_paths = [
        image_path.with_suffix('.txt'),
        image_path.with_suffix('.caption'),
        image_path.parent / f"{image_path.stem}.txt",
    ]
    
    for txt_path in txt_paths:
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    
    return None


def worker_process(rank: int, world_size: int, args, image_paths: List[Path],
                   output_dir: Path, input_root: Path, progress_queue):
    """多卡 worker 进程"""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    
    # 加载文本编码器
    print(f"[GPU {rank}] Loading Text Encoder...", flush=True)
    
    model_path = args.text_encoder
    tokenizer_path = model_path
    
    # 检查 tokenizer 路径
    path_obj = Path(model_path)
    if not (path_obj / "tokenizer.json").exists() and (path_obj.parent / "tokenizer").exists():
        tokenizer_path = str(path_obj.parent / "tokenizer")
        logger.info(f"Tokenizer not found in {model_path}, using {tokenizer_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    model.to(device)
    model.eval()
    print(f"[GPU {rank}] Text Encoder loaded, processing {len(image_paths)} images", flush=True)
    
    processed = 0
    skipped = 0
    
    for i, image_path in enumerate(image_paths):
        name = image_path.stem
        
        # 计算输出路径
        try:
            rel_path = image_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
        
        output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
        
        if args.skip_existing and output_file.exists():
            skipped += 1
            progress_queue.put(1)
            continue
        
        # 获取 caption
        caption = get_caption(image_path)
        if caption is None:
            logger.warning(f"No caption found for {image_path}")
            progress_queue.put(1)
            continue
        
        try:
            # Tokenize
            inputs = tokenizer(
                caption,
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
                padding="max_length",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Encode
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                embed = hidden_states.squeeze(0).to(dtype=dtype)
            
            # 保存
            target_dir.mkdir(parents=True, exist_ok=True)
            sd = {"varlen_vl_embed_bf16": embed.cpu()}
            save_file(sd, str(output_file))
            
            processed += 1
            progress_queue.put(1)
            
        except Exception as e:
            print(f"[GPU {rank}] Error: {image_path.name}: {e}", flush=True)
            progress_queue.put(1)
    
    # 清理
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"[GPU {rank}] Completed: {processed} processed, {skipped} skipped", flush=True)


def progress_monitor(total: int, progress_queue, done_event):
    """进度监控进程"""
    count = 0
    while not done_event.is_set() or not progress_queue.empty():
        try:
            delta = progress_queue.get(timeout=0.5)
            count += delta
            print(f"Progress: {count}/{total}", flush=True)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Cache text embeddings for Z-Image training (Multi-GPU)")
    parser.add_argument("--text_encoder", type=str, required=True, help="Text encoder path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs (0=auto)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = Path(args.input_dir)
    
    # 查找图片
    images = find_images(args.input_dir)
    total = len(images)
    print(f"Found {total} images", flush=True)
    
    if total == 0:
        print("No images to process", flush=True)
        return
    
    # 检测 GPU 数量
    if args.num_gpus > 0:
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
    else:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        # 单卡模式
        print(f"Using single GPU mode", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        
        print(f"Loading Text Encoder: {args.text_encoder}", flush=True)
        
        model_path = args.text_encoder
        tokenizer_path = model_path
        
        path_obj = Path(model_path)
        if not (path_obj / "tokenizer.json").exists() and (path_obj.parent / "tokenizer").exists():
            tokenizer_path = str(path_obj.parent / "tokenizer")
            logger.info(f"Tokenizer not found in {model_path}, using {tokenizer_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        model.to(device)
        model.eval()
        print("Text Encoder loaded successfully", flush=True)
        
        processed = 0
        skipped = 0
        
        for i, image_path in enumerate(images, 1):
            name = image_path.stem
            
            try:
                rel_path = image_path.relative_to(input_root)
                target_dir = output_dir / rel_path.parent
            except ValueError:
                target_dir = output_dir
            
            output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
            
            if args.skip_existing and output_file.exists():
                skipped += 1
                print(f"Progress: {i}/{total}", flush=True)
                continue
            
            caption = get_caption(image_path)
            if caption is None:
                logger.warning(f"No caption found for {image_path}")
                print(f"Progress: {i}/{total}", flush=True)
                continue
            
            try:
                inputs = tokenizer(
                    caption,
                    return_tensors="pt",
                    max_length=args.max_length,
                    truncation=True,
                    padding="max_length",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    embed = hidden_states.squeeze(0).to(dtype=dtype)
                
                target_dir.mkdir(parents=True, exist_ok=True)
                sd = {"varlen_vl_embed_bf16": embed.cpu()}
                save_file(sd, str(output_file))
                
                processed += 1
                print(f"Progress: {i}/{total}", flush=True)
                
            except Exception as e:
                print(f"Error: {image_path}: {e}", flush=True)
                print(f"Progress: {i}/{total}", flush=True)
        
        print(f"Text encoding completed! Processed: {processed}, Skipped: {skipped}", flush=True)
        
        del model
        del tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Text Encoder unloaded, GPU memory released", flush=True)
    
    else:
        # 多卡模式
        print(f"🚀 Multi-GPU mode: using {num_gpus} GPUs", flush=True)
        print(f"Progress: 0/{total}", flush=True)
        
        # 分片
        chunk_size = (total + num_gpus - 1) // num_gpus
        chunks = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            if start < total:
                chunks.append(images[start:end])
        
        print(f"Distributing {total} images across {num_gpus} GPUs", flush=True)
        for i, chunk in enumerate(chunks):
            print(f"  GPU {i}: {len(chunk)} images", flush=True)
        
        # 创建进度队列和完成事件
        progress_queue = mp.Queue()
        done_event = mp.Event()
        
        # 启动进度监控
        import threading
        monitor_thread = threading.Thread(
            target=progress_monitor,
            args=(total, progress_queue, done_event),
            daemon=True
        )
        monitor_thread.start()
        
        # 启动多卡处理
        processes = []
        for rank, chunk in enumerate(chunks):
            p = mp.Process(
                target=worker_process,
                args=(rank, num_gpus, args, chunk, output_dir, input_root, progress_queue)
            )
            p.start()
            processes.append(p)
        
        # 等待完成
        for p in processes:
            p.join()
        
        # 通知进度监控结束
        done_event.set()
        monitor_thread.join(timeout=2)
        
        print(f"Multi-GPU text encoding completed!", flush=True)


if __name__ == "__main__":
    # 强制使用 spawn 模式避免 CUDA 冲突
    mp.set_start_method('spawn', force=True)
    main()
