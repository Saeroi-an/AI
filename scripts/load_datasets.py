from datasets import load_dataset
import shutil
import os

# cord-v2 데이터셋 다운로드 (샘플만 1~2개 사용)
dataset = load_dataset("naver-clova-ix/cord-v2", split="train[:10]")  # 처음 2개 샘플만

# 저장할 경로
img_dir = "data/cord_sample/images"
ann_dir = "data/cord_sample/annotations"

os.makedirs(img_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)

# 샘플 이미지와 JSON annotation 저장
for i, sample in enumerate(dataset):
    # 이미지 저장
    img_path = os.path.join(img_dir, f"{i:05d}.jpg")
    sample["image"].save(img_path)

    # JSON annotation 저장
    ann_path = os.path.join(ann_dir, f"{i:05d}.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        import json
        json.dump(sample["ground_truth"], f, ensure_ascii=False, indent=2)

print("CORD 샘플 데이터 준비 완료!")
