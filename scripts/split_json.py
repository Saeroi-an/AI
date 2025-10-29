import json
import os
from pathlib import Path

# ----------------------------
# 경로 설정
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # scripts/에서 한 단계 위
ORIGINAL_JSON = BASE_DIR / "synth_rx/zh_cord_LLaVA_datasets_final2.json"

TRAIN_IMAGES_DIR = BASE_DIR / "data/ko_zh_datasets_2/train"
TEST_IMAGES_DIR  = BASE_DIR / "data/ko_zh_datasets_2/test"

TRAIN_JSON = BASE_DIR / "data/ko_zh_datasets_2/train_zh_ko.json"
TEST_JSON  = BASE_DIR / "data/ko_zh_datasets_2/test_zh_ko.json"

# ----------------------------
# train/test 폴더에 있는 이미지 파일명 추출
# ----------------------------
train_images = set(f.name for f in TRAIN_IMAGES_DIR.glob("*.jpg"))
test_images  = set(f.name for f in TEST_IMAGES_DIR.glob("*.jpg"))

print(f"Number of train images found: {len(train_images)}")
print(f"Number of test images found: {len(test_images)}")

# ----------------------------
# 기존 JSON 읽기
# ----------------------------
if not ORIGINAL_JSON.exists():
    raise FileNotFoundError(f"{ORIGINAL_JSON} 파일이 존재하지 않습니다.")

with open(ORIGINAL_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# ----------------------------
# train/test JSON 생성
# ----------------------------
train_data = []
test_data  = []

for entry in data:
    image_name = Path(entry["image"]).name  # 파일명만 추출
    if image_name in train_images:
        train_data.append(entry)
    elif image_name in test_images:
        test_data.append(entry)
    else:
        # train/test 폴더에 없는 이미지
        print(f"Warning: {image_name} not found in train/test folders, skipping")

# ----------------------------
# JSON 저장
# ----------------------------
TRAIN_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(TRAIN_JSON, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(TEST_JSON, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"Train JSON entries: {len(train_data)}")
print(f"Test JSON entries: {len(test_data)}")
