#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ollama 图片标注脚本 - 严格参照 NONE-Ollama-Tagging.py
"""
import argparse
import base64
import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image

# 默认提示词
DEFAULT_PROMPT = """Describe this image in detail for AI image generation training. 
Focus on: subject, clothing, pose, expression, hair, accessories.
Output format: comma-separated tags.
Be concise and specific."""


def parse_args():
    parser = argparse.ArgumentParser(description="Ollama 图片标注工具")
    parser.add_argument("--input_dir", type=str, required=True, help="图片目录")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Ollama API 地址")
    parser.add_argument("--model", type=str, default="llava:13b", help="视觉模型名称")
    parser.add_argument("--prompt", type=str, default=None, help="自定义提示词")
    parser.add_argument("--trigger_word", type=str, default="", help="触发词（添加到标注开头）")
    parser.add_argument("--max_long_edge", type=int, default=512, help="图片最长边限制")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="跳过已标注图片")
    parser.add_argument("--timeout", type=int, default=180, help="API 超时时间（秒）")
    parser.add_argument("--delay", type=float, default=0.2, help="请求间隔（秒）")
    return parser.parse_args()


def get_image_paths(input_dir: Path, skip_existing: bool = True) -> list:
    """获取待标注图片列表"""
    IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_paths = []
    
    for ext in IMAGE_EXTENSIONS:
        for img_path in input_dir.rglob(ext):
            if skip_existing and img_path.with_suffix(".txt").exists():
                continue
            image_paths.append(img_path)
    
    return sorted(image_paths)


def resize_for_api(img_path: Path, max_long_edge: int = 512) -> bytes:
    """缩放图片并返回 JPEG 字节 - 完全参照用户脚本"""
    with Image.open(img_path) as im:
        # 转换为 RGB 模式
        if im.mode in ('RGBA', 'P'):
            im = im.convert('RGB')
        
        w, h = im.size
        long_edge = max(w, h)
        
        # 缩放到最长边限制
        if long_edge > max_long_edge:
            ratio = max_long_edge / long_edge
            new_size = (int(w * ratio), int(h * ratio))
            im = im.resize(new_size, Image.BICUBIC)
        
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        return buf.getvalue()


def generate_caption(img_path: Path, ollama_url: str, model: str, prompt: str, 
                     max_long_edge: int, timeout: int) -> str | None:
    """生成单张图片描述 - 完全参照用户脚本逻辑"""
    try:
        jpeg_bytes = resize_for_api(img_path, max_long_edge)
        base64_img = base64.b64encode(jpeg_bytes).decode()
        
        # 完全按照用户脚本的 payload，不加任何额外参数
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [base64_img],
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        
        response_json = resp.json()
        if "response" in response_json:
            caption = response_json["response"].strip()
            # 简单清洗
            caption = caption.replace("```markdown", "").replace("```", "").strip()
            return caption if caption else None
        else:
            print(f"[ERROR] API response missing 'response' field", flush=True)
            return None
            
    except requests.exceptions.Timeout:
        print(f"[ERROR] Timeout: {img_path.name}", flush=True)
        return None
    except Exception as e:
        print(f"[ERROR] Failed ({img_path.name}): {e}", flush=True)
        return None


def save_caption(img_path: Path, caption: str, trigger_word: str = ""):
    """保存标注文件"""
    if trigger_word.strip():
        caption = f"{trigger_word.strip()}, {caption}"
    txt_path = img_path.with_suffix(".txt")
    txt_path.write_text(caption, encoding="utf-8")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"[ERROR] Directory not found: {input_dir}", flush=True)
        print("Progress: 0/0", flush=True)
        sys.exit(1)
    
    prompt = args.prompt if args.prompt else DEFAULT_PROMPT
    image_paths = get_image_paths(input_dir, args.skip_existing)
    
    if not image_paths:
        print("[INFO] All images already tagged", flush=True)
        print("Progress: 0/0", flush=True)
        return
    
    total = len(image_paths)
    print(f"[INFO] Images to tag: {total}", flush=True)
    print(f"[INFO] Model: {args.model}", flush=True)
    print(f"[INFO] API: {args.ollama_url}", flush=True)
    print(f"Progress: 0/{total}", flush=True)
    sys.stdout.flush()
    
    for i, img_path in enumerate(image_paths):
        print(f"[FILE] {img_path.name}", flush=True)
        sys.stdout.flush()
        
        caption = generate_caption(
            img_path,
            args.ollama_url,
            args.model,
            prompt,
            args.max_long_edge,
            args.timeout
        )
        
        if caption:
            save_caption(img_path, caption, args.trigger_word)
            print(f"[OK] {img_path.name}", flush=True)
        else:
            print(f"[FAIL] {img_path.name}", flush=True)
        
        print(f"Progress: {i+1}/{total}", flush=True)
        sys.stdout.flush()
        
        if args.delay > 0 and i < total - 1:
            time.sleep(args.delay)
    
    print("[INFO] Tagging complete", flush=True)
    print(f"Progress: {total}/{total}", flush=True)


if __name__ == "__main__":
    main()
