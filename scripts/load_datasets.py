from datasets import Dataset, DatasetDict, Features, Value, Image
from pathlib import Path

# ----------------------------
# 경로 설정
# ----------------------------
BASE_DIR = Path("Qwen2-vl-finetune-wo/data/ko_zh_datasets_2")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR  = BASE_DIR / "test"

def build_dataset(image_dir):
    # 이미지 파일 리스트
    image_paths = list(image_dir.glob("*.jpg"))
    
    # 데이터 생성: {"filename": "0001_ko.jpg", "image": "/full/path/0001_ko.jpg"}
    data = [{"filename": img.name, "image": str(img)} for img in image_paths]
    
    # features 정의
    features = Features({
        "filename": Value("string"),
        "image": Image()
    })
    
    return Dataset.from_list(data, features=features)

# ----------------------------
# train/test Dataset 생성
# ----------------------------
train_dataset = build_dataset(TRAIN_DIR)
test_dataset  = build_dataset(TEST_DIR)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# ----------------------------
# HuggingFace Hub 업로드
# ----------------------------
dataset_dict.push_to_hub("Rfy23/ko_zh_dataset_v1", token="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")
