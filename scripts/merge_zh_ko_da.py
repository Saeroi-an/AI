import json
import shutil
from pathlib import Path

# ----------------------------
# 경로 설정
# ----------------------------

# 새 이미지 저장 폴더
new_img_dir = Path("data/ko_zh_datasets")
new_img_dir.mkdir(parents=True, exist_ok=True)

# 1️⃣ _ko 관련
ko_json_in = Path("synth_rx/final_qa_datasets_LLaVA_for_finetune_updated_del_aug.json")
ko_json_out = Path("synth_rx/final_qa_datasets_LLaVA_for_finetune_updated_del_aug_ko.json")
ko_img_dir = Path("data/cord_sample/images")

# 2️⃣ _zh 관련
zh_json_in = Path("synth_rx/llava_rx_dataset_prescriptions_zh.json")
zh_json_out = Path("synth_rx/llava_rx_dataset_prescriptions_zh_zh.json")
zh_img_dir = Path("data/synth_prescriptions/images")

# ----------------------------
# 처리 함수
# ----------------------------
def process_json(json_in_path, json_out_path, orig_img_dir, suffix):
    """
    JSON 데이터를 불러와 id와 image에 접미사 추가,
    이미지 파일 복사 및 새 폴더에 저장
    """
    with open(json_in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # 1️⃣ id에 접미사 추가
        orig_id = item["id"]
        item["id"] = f"{orig_id}_{suffix}"

        # 2️⃣ image 경로 및 파일명 처리
        orig_image_name = Path(item["image"]).stem
        ext = Path(item["image"]).suffix

        new_image_name = f"{orig_image_name}_{suffix}{ext}"
        new_image_path = new_img_dir / new_image_name
        item["image"] = str(new_image_path)

        # 3️⃣ 실제 이미지 복사
        src_image_path = orig_img_dir / f"{orig_image_name}{ext}"
        if not src_image_path.exists():
            print(f"⚠️ 원본 이미지 없음: {src_image_path}")
            continue
        shutil.copy(src_image_path, new_image_path)

    # 4️⃣ 새 JSON 파일 저장
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 처리 완료: {json_out_path}, 이미지 복사 완료 ({len(data)}장)")

# ----------------------------
# 실제 처리
# ----------------------------
# 한국어 _ko
process_json(ko_json_in, ko_json_out, ko_img_dir, "ko")

# 중국어 _zh
process_json(zh_json_in, zh_json_out, zh_img_dir, "zh")
