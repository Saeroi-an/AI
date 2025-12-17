import os
import json
import cv2
import albumentations as A
from tqdm import tqdm

# === 경로 설정 ===
IMG_DIR = "data/ko_zh_datasets_3/test"
JSON_PATH = "data/ko_zh_datasets_3/test_zh_ko.json"

# === Albumentations 변환 정의 ===
# aug_color = A.Compose([
#     A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.7),
#     A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=5, val_shift_limit=5, p=0.5),
#     A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01, p=0.5),

#
#     A.GaussNoise(var_limit=(3, 10), mean=0, p=0.4),

#     
#     A.ImageCompression(quality_lower=85, quality_upper=95, p=0.3)
# ])

aug_blur = A.Compose([
    
    A.MotionBlur(blur_limit=(1, 3), p=0.5),
    
  
    A.GaussianBlur(blur_limit=(1, 3), p=0.5),

    
    # A.GaussNoise(var_limit=(3, 10), mean=0, p=0.4),

   
    A.ImageCompression(quality_lower=85, quality_upper=95, p=0.3)
])


aug_distort = A.Compose([
    A.Affine(rotate=(-10, 10), shear=(-5, 5), scale=(0.9, 1.1), p=1.0),
    A.Perspective(scale=(0.02, 0.05), p=1.0)
])

augmentations = {
    "_aug1": aug_blur,
    "_aug2": aug_distort
}

# === JSON 불러오기 ===
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# === _zh 이미지만 찾기 ===
zh_images = [f for f in os.listdir(IMG_DIR) if f.endswith("_zh.jpg")]

print(f"Found {len(zh_images)} _zh images.")

# === 이미지 증강 및 저장 ===
for img_name in tqdm(zh_images):
    img_path = os.path.join(IMG_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Cannot read image: {img_name}")
        continue

    # 변환 수행
    for suffix, transform in augmentations.items():
        aug_img = transform(image=image)["image"]
        out_name = img_name.replace(".jpg", f"{suffix}.jpg")
        out_path = os.path.join(IMG_DIR, out_name)
        cv2.imwrite(out_path, aug_img)

# === JSON 갱신 ===
new_entries = []

for entry in data:
    if entry["id"].endswith("_zh"):
        base_id = entry["id"]
        base_img = entry["image"]
        for suffix in ["_aug1", "_aug2"]:
            new_entry = {
                "id": f"{base_id}{suffix}",
                "image": base_img.replace(".jpg", f"{suffix}.jpg"),
                "conversations": entry["conversations"]
            }
            new_entries.append(new_entry)

# 기존 데이터 + 새 데이터 합치기
data.extend(new_entries)

# === JSON 저장 ===
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 완료! {len(new_entries)}개의 증강된 JSON 항목이 추가되었습니다.")
print("✅ 이미지 증강 및 저장 완료.")
