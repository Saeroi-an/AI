import json
import os
from glob import glob

# 원본 json 파일 경로
input_folder = "data/cord_sample/annotations"  # 기존 json 파일이 있는 폴더
output_folder = "data/cord_sample/annotations_json"  # 저장할 폴더

os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 json 파일 읽기
json_files = glob(os.path.join(input_folder, "*.json"))

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        raw_data = f.read()
        
        # 문자열로 된 JSON이면 두 번 파싱
        try:
            data = json.loads(raw_data)
            if isinstance(data, str):
                data = json.loads(data)
        except json.JSONDecodeError as e:
            print(f"Failed to parse {json_file}: {e}")
            continue

    # output 파일 경로
    base_name = os.path.basename(json_file)
    output_path = os.path.join(output_folder, base_name)

    # 그대로 다시 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {output_path}")
