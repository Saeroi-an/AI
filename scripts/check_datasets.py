import os
import json
from pathlib import Path

# ----- 수정 -----
JSON_PATH = "/home/jwlee/volume/Qwen2-vl-finetune-wo/data/ko_zh_datasets_4/val_zh_ko.json"
TOTAL_IMAGE_DIR = "/home/jwlee/volume/Qwen2-vl-finetune-wo/data/ko_zh_datasets_4/val"
# -----------------

missing = []
total = 0

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)   # JSONL이 아니라 list임

for item in data:
    img_field = item.get("image", "")  # ex: "data/ko_zh_datasets_4/total_images/00427_zh.jpg"

    # 파일 이름만 추출
    filename = os.path.basename(img_field)

    total += 1
    full_path = os.path.join(TOTAL_IMAGE_DIR, filename)

    if not os.path.exists(full_path):
        missing.append(full_path)

print(f"Checked {total} images")
print(f"Missing: {len(missing)}")

if missing:
    print("\n--- Missing files ---")
    for m in missing[:50]:
        print(m)
