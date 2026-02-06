#!/usr/bin/env python3
"""
将训练好的 LoRA 转换为 ComfyUI 兼容格式

ComfyUI Z-Image 使用合并的 qkv，需要将分离的 to_q/to_k/to_v LoRA 合并

参考: https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/z_image_convert_original_to_comfy.py

合并逻辑 (按照官方脚本的顺序: q, k, v):
  qkv.weight = cat([q.weight, k.weight, v.weight], dim=0)  # [3*out, in]

对于 LoRA:
  - lora_down: [rank, in] -> qkv_down = cat([q,k,v], dim=0) = [3*rank, in]
  - lora_up: [out, rank] -> qkv_up = block_diag([q,k,v]) = [3*out, 3*rank]
"""

import argparse
from pathlib import Path
from collections import defaultdict
import torch
from safetensors.torch import load_file, save_file


# ComfyUI 键名替换规则 (from official script)
COMFYUI_REPLACE_KEYS = {
    ".attention.to_out.0.": ".attention.out.",
    ".attention.to_out.": ".attention.out.",  # 无 .0 的情况
    ".attention.norm_k.": ".attention.k_norm.",
    ".attention.norm_q.": ".attention.q_norm.",
    "all_final_layer.2-1.": "final_layer.",
    "all_x_embedder.2-1.": "x_embedder.",
}


def merge_qkv_lora(q_down, k_down, v_down, q_up, k_up, v_up, q_alpha=None):
    """
    合并 q/k/v LoRA 为 qkv LoRA (数学等价)
    
    输入 shapes (PyTorch Linear convention):
      - lora_down.weight: [rank, in_dim]
      - lora_up.weight: [out_dim, rank]
    
    合并策略 (保持数学等价):
      - qkv_down = cat([q_down, k_down, v_down], dim=0)  # [3*rank, in_dim]
      - qkv_up = block_diag([q_up, k_up, v_up])          # [3*out_dim, 3*rank]
    
    这样保证: x @ qkv_down.T @ qkv_up.T = cat([x@q_down.T@q_up.T, ...], dim=-1)
    """
    rank = q_down.shape[0]
    in_dim = q_down.shape[1]
    out_dim = q_up.shape[0]
    dtype = q_down.dtype
    device = q_down.device
    
    # qkv_down: concat along dim=0 (output/rank dimension)
    # [3*rank, in_dim]
    qkv_down = torch.cat([q_down, k_down, v_down], dim=0)
    
    # qkv_up: block diagonal
    # [3*out_dim, 3*rank]
    qkv_up = torch.zeros(3 * out_dim, 3 * rank, dtype=dtype, device=device)
    qkv_up[0:out_dim, 0:rank] = q_up
    qkv_up[out_dim:2*out_dim, rank:2*rank] = k_up
    qkv_up[2*out_dim:3*out_dim, 2*rank:3*rank] = v_up
    
    # alpha: 使用 q 的 alpha (通常都一样)
    # 由于 rank 变成 3 倍，alpha 也应该变成 3 倍以保持相同的缩放
    merged_alpha = q_alpha * 3 if q_alpha is not None else None
    
    return qkv_down, qkv_up, merged_alpha


def convert_key_to_comfyui(key: str) -> str:
    """转换单个键名为 ComfyUI 格式"""
    # 移除 diffusion_model. 前缀
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model."):]
    
    # 应用替换规则
    for old, new in COMFYUI_REPLACE_KEYS.items():
        key = key.replace(old, new)
    
    return key


def to_comfyui_lora_key(base_path: str, lora_part: str) -> str:
    """
    生成 ComfyUI LoRA 键名
    
    base_path: layers.0.attention.qkv
    lora_part: lora_down, lora_up, alpha
    
    -> lora_unet_layers_0_attention_qkv.lora_down.weight
    """
    # 路径用下划线连接
    base_key = base_path.replace(".", "_")
    
    if lora_part == "alpha":
        return f"lora_unet_{base_key}.alpha"
    else:
        return f"lora_unet_{base_key}.{lora_part}.weight"


def convert_to_comfyui(input_path: str, output_path: str, dtype: str = None):
    """转换为 ComfyUI 格式，合并 qkv"""
    print(f"Loading: {input_path}")
    state_dict = load_file(input_path)
    
    # 数据类型
    cast_to = None
    if dtype == "fp16":
        cast_to = torch.float16
    elif dtype == "bf16":
        cast_to = torch.bfloat16
    elif dtype == "fp8":
        cast_to = torch.float8_e4m3fn
    
    # 分组收集 q/k/v LoRA
    # key: base_path (e.g., "layers.0.attention")
    # value: {"q_lora_down": tensor, "k_lora_up": tensor, ...}
    qkv_groups = defaultdict(dict)
    other_tensors = {}
    
    for key, value in state_dict.items():
        # 清理前缀
        clean_key = key
        if clean_key.startswith("diffusion_model."):
            clean_key = clean_key[len("diffusion_model."):]
        
        # 检测 q/k/v (支持两种格式: .to_q. 和 .to.q.)
        for qkv_type in ["to_q", "to_k", "to_v"]:
            marker = f".attention.{qkv_type}."
            marker_old = f".attention.{qkv_type.replace('_', '.')}."  # 旧格式 .to.q.
            
            actual_marker = None
            if marker in clean_key:
                actual_marker = marker
            elif marker_old in clean_key:
                actual_marker = marker_old
            
            if actual_marker:
                # 提取 base_path 和 lora_part
                # e.g., "layers.0.attention.to_q.lora_down.weight"
                # -> base = "layers.0.attention", part = "lora_down"
                base_path = clean_key.split(actual_marker)[0] + ".attention"
                suffix = clean_key.split(actual_marker)[1]  # "lora_down.weight" or "alpha"
                
                if suffix.endswith(".weight"):
                    lora_part = suffix.replace(".weight", "")  # "lora_down" or "lora_up"
                else:
                    lora_part = suffix  # "alpha"
                
                # 简化键名: q_lora_down, k_lora_up, etc.
                short_type = qkv_type.replace("to_", "")  # "q", "k", "v"
                group_key = f"{short_type}_{lora_part}"
                
                qkv_groups[base_path][group_key] = value
                break
        else:
            # 非 qkv 的权重
            other_tensors[clean_key] = value
    
    new_state_dict = {}
    merged_count = 0
    
    # 合并 qkv
    for base_path, parts in qkv_groups.items():
        required = ["q_lora_down", "k_lora_down", "v_lora_down",
                    "q_lora_up", "k_lora_up", "v_lora_up"]
        
        if all(k in parts for k in required):
            # 完整的 qkv，合并
            q_alpha = parts.get("q_alpha")
            qkv_down, qkv_up, merged_alpha = merge_qkv_lora(
                parts["q_lora_down"], parts["k_lora_down"], parts["v_lora_down"],
                parts["q_lora_up"], parts["k_lora_up"], parts["v_lora_up"],
                q_alpha
            )
            
            # 应用键名替换
            comfy_base = convert_key_to_comfyui(base_path) + ".qkv"
            
            new_state_dict[to_comfyui_lora_key(comfy_base, "lora_down")] = qkv_down
            new_state_dict[to_comfyui_lora_key(comfy_base, "lora_up")] = qkv_up
            if merged_alpha is not None:
                new_state_dict[to_comfyui_lora_key(comfy_base, "alpha")] = merged_alpha
            
            merged_count += 1
            if merged_count <= 3:
                print(f"  Merged: {base_path}.qkv")
                print(f"    down: {tuple(qkv_down.shape)}, up: {tuple(qkv_up.shape)}")
        else:
            # 不完整，保持分离（fallback）
            print(f"  Warning: Incomplete qkv at {base_path}, keeping separate")
            for part_name, tensor in parts.items():
                qkv_type, lora_part = part_name.split("_", 1)
                comfy_base = convert_key_to_comfyui(base_path) + f".to_{qkv_type}"
                new_state_dict[to_comfyui_lora_key(comfy_base, lora_part)] = tensor
    
    if merged_count > 3:
        print(f"  ... and {merged_count - 3} more qkv merges")
    
    # 处理其他键 (to_out, feed_forward, etc.)
    for key, value in other_tensors.items():
        comfy_key = convert_key_to_comfyui(key)
        
        # 解析 lora 部分
        if ".lora_down.weight" in comfy_key:
            base = comfy_key.replace(".lora_down.weight", "")
            new_key = to_comfyui_lora_key(base, "lora_down")
        elif ".lora_up.weight" in comfy_key:
            base = comfy_key.replace(".lora_up.weight", "")
            new_key = to_comfyui_lora_key(base, "lora_up")
        elif ".alpha" in comfy_key:
            base = comfy_key.replace(".alpha", "")
            new_key = to_comfyui_lora_key(base, "alpha")
        else:
            new_key = f"lora_unet_{comfy_key.replace('.', '_')}"
        
        new_state_dict[new_key] = value
    
    # 类型转换
    if cast_to is not None:
        for key in new_state_dict:
            if hasattr(new_state_dict[key], 'to'):
                new_state_dict[key] = new_state_dict[key].to(cast_to)
    
    # 保存
    save_file(new_state_dict, output_path)
    
    print(f"\n✅ Saved: {output_path}")
    print(f"   Total keys: {len(new_state_dict)}")
    print(f"   Merged qkv: {merged_count}")
    print(f"\n⚠️  Note: qkv merge increases rank 3x (down: [3r, in], up: [3o, 3r])")


def inspect_lora(input_path: str):
    """检查 LoRA 文件格式"""
    print(f"Inspecting: {input_path}")
    state_dict = load_file(input_path)
    
    print(f"\nTotal keys: {len(state_dict)}")
    
    # 统计
    stats = {"to_q": 0, "to_k": 0, "to_v": 0, "to_out": 0, "qkv": 0, "feed_forward": 0, "other": 0}
    for key in state_dict.keys():
        matched = False
        for pattern in ["to_q", "to_k", "to_v", "to_out", "qkv", "feed_forward"]:
            if pattern in key:
                stats[pattern] += 1
                matched = True
                break
        if not matched:
            stats["other"] += 1
    
    print("\nLayer distribution:")
    for k, v in stats.items():
        if v > 0:
            print(f"  {k}: {v}")
    
    print("\nSample keys (first 20):")
    for i, (key, val) in enumerate(sorted(state_dict.items())):
        if i >= 20:
            print(f"  ... and {len(state_dict) - 20} more")
            break
        shape = tuple(val.shape) if hasattr(val, 'shape') else type(val).__name__
        print(f"  {key}: {shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image LoRA -> ComfyUI 格式转换 (合并 qkv)"
    )
    parser.add_argument("input", help="输入 LoRA 文件")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp8"], help="数据类型")
    parser.add_argument("--inspect", action="store_true", help="仅检查格式")
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_lora(args.input)
        return
    
    if not args.output:
        p = Path(args.input)
        args.output = str(p.with_stem(p.stem + "_comfyui"))
    
    convert_to_comfyui(args.input, args.output, args.dtype)


if __name__ == "__main__":
    main()
