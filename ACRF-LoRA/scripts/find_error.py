import os

dataset_dir = "/mnt/sda/hf/qwen-VL/Z-Image/None_Z-image-Turbo_trainer/datasets/fashion-dataset"

# 提取所有文件的基础ID（去掉后缀和尺寸）
ids = set()
# 1. 扫描所有safetensors文件，提取基础ID
for fname in os.listdir(dataset_dir):
    if "_zi.safetensors" in fname:
        # 去掉尺寸和后缀，比如"10079_768x1024_zi.safetensors" → "10079"
        base_id = fname.split("_")[0]
        ids.add(base_id)

# 2. 检查每个ID是否有配套的jpg和txt
invalid_ids = []
for base_id in ids:
    has_jpg = os.path.exists(os.path.join(dataset_dir, f"{base_id}.jpg"))
    has_txt = os.path.exists(os.path.join(dataset_dir, f"{base_id}.txt"))
    has_safetensors = any(
        base_id in fname and "_zi.safetensors" in fname 
        for fname in os.listdir(dataset_dir)
    )
    
    if not (has_jpg and has_txt and has_safetensors):
        invalid_ids.append({
            "id": base_id,
            "missing_jpg": not has_jpg,
            "missing_txt": not has_txt,
            "missing_safetensors": not has_safetensors
        })

# 输出结果
if invalid_ids:
    print(f"找到 {len(invalid_ids)} 个无效ID（缺少配套文件）：")
    for item in invalid_ids[:10]:  # 只打印前10个，避免刷屏
        print(f"ID: {item['id']} - 缺jpg: {item['missing_jpg']}, 缺txt: {item['missing_txt']}")
else:
    print("所有ID都有完整的jpg/txt/safetensors配套文件")

# 3. 统计有效文件数量
valid_count = len(ids) - len(invalid_ids)
print(f"\n有效ID数量：{valid_count}")
if valid_count == 0:
    print("⚠️  没有任何有效ID，这是导致报错的直接原因！")