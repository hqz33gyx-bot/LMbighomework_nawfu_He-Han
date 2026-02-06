import csv
import os
from pathlib import Path

csv_file_path = "datasets/fashion-dataset/styles.csv"
output_dir = Path(csv_file_path).parent / "style_txt_files"
output_dir.mkdir(exist_ok=True)

def process_csv_row(row, header):
    """处理单行数据，生成连贯语句并写入txt文件"""
    row_id = row.get("id", "").strip()
    if not row_id:
        print("跳过空ID的行")
        return
    
    gender = row.get("gender", "").strip() or "unknown"
    master_category = row.get("masterCategory", "").strip() or "unknown"
    sub_category = row.get("subCategory", "").strip() or "unknown"
    article_type = row.get("articleType", "").strip() or "unknown"
    base_colour = row.get("baseColour", "").strip() or "unknown"
    season = row.get("season", "").strip() or "unknown"
    year = row.get("year", "").strip() or "unknown"
    usage = row.get("usage", "").strip() or "unknown"
    product_name = row.get("productDisplayName", "").strip() or "unknown"
    
    # 拼接成连贯的一句话
    sentence = (
        f"This is a {gender} oriented {master_category} product, belonging to the {sub_category} category, "
        f"specifically an {article_type} in {base_colour} colour. "
        f"It is designed for the {season} season of {year} and is intended for {usage}. "
        f"The product display name is: {product_name}."
    )
    
    txt_file_path = output_dir / f"{row_id}.txt"
    
    try:
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(sentence)
        print(f"成功生成文件: {txt_file_path}")
    except Exception as e:
        print(f"生成文件{row_id}.txt失败: {str(e)}")

def main():
    if not os.path.exists(csv_file_path):
        print(f"错误：CSV文件不存在 - {csv_file_path}")
        return
    
    try:
        with open(csv_file_path, "r", encoding="utf-8") as csv_file:

            csv_reader = csv.DictReader(csv_file)
            header = csv_reader.fieldnames 

            for row_num, row in enumerate(csv_reader, start=2): 
                print(f"处理第{row_num}行，ID: {row.get('id')}")
                process_csv_row(row, header)
    except UnicodeDecodeError:
        with open(csv_file_path, "r", encoding="gbk") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            header = csv_reader.fieldnames
            for row_num, row in enumerate(csv_reader, start=2):
                print(f"处理第{row_num}行，ID: {row.get('id')}")
                process_csv_row(row, header)
    except Exception as e:
        print(f"读取CSV文件失败: {str(e)}")

if __name__ == "__main__":
    main()
    print("=== 处理完成 ===")