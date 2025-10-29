import os
import shutil
from pathlib import Path
import random

# ----------------------------
# 절대경로 기준 설정
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # scripts/ -> Qwen2-vl-finetune-wo/
DATA_DIR = BASE_DIR / "data/ko_zh_datasets"

# 새 폴더
NEW_DATA_DIR = BASE_DIR / "data/ko_zh_datasets_2"
TRAIN_DIR = NEW_DATA_DIR / "train"
TEST_DIR  = NEW_DATA_DIR / "test"

# train/test 폴더 생성
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# _ko, _zh 기준으로 파일 나누기
# ----------------------------
ko_files = sorted(DATA_DIR.glob("*_ko.jpg"))
zh_files = sorted(DATA_DIR.glob("*_zh.jpg"))

print(f"Found _ko files: {len(ko_files)}")
print(f"Found _zh files: {len(zh_files)}")

# 랜덤 섞기
random.seed(42)
random.shuffle(ko_files)
random.shuffle(zh_files)

# _ko: 600 train / 101 test
ko_train_files = ko_files[:600]
ko_test_files  = ko_files[600:701]

# _zh: 1000 train / 300 test
zh_train_files = zh_files[:1000]
zh_test_files  = zh_files[1000:1300]

# ----------------------------
# 파일 복사 함수
# ----------------------------
def copy_files(file_list, dest_dir):
    for f in file_list:
        shutil.copy(f, dest_dir / f.name)

# train/test 폴더에 복사
copy_files(ko_train_files, TRAIN_DIR)
copy_files(zh_train_files, TRAIN_DIR)
copy_files(ko_test_files, TEST_DIR)
copy_files(zh_test_files, TEST_DIR)

# 결과 출력
print(f"Train: {len(ko_train_files) + len(zh_train_files)} images")
print(f"Test:  {len(ko_test_files) + len(zh_test_files)} images")
