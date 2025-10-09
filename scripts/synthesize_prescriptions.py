import json
import random
import os

# 필드 JSON 파일 위치
FIELD_FILES = {
    "prescription_type": "scripts/fields/prescription_type.json",
    "patient_info": "scripts/fields/patient_info.json",
    "diagnosis_code": "scripts/fields/diagnosis_code.json",
    "disease_name": "scripts/fields/disease_name.json",
    "current_medication": "scripts/fields/current_medication.json",
    "dosage_info": "scripts/fields/dosage_info.json",
    "usage_period": "scripts/fields/usage_period.json",
    "insurance": "scripts/fields/insurance.json",
    "hospital_info": "scripts/fields/hospital_info.json"
}

NUM_SAMPLES = 10
SAVE_PATH = "data/synth_rx/prescriptions.jsonl"

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# 각 필드별 값 불러오기
fields_data = {}
for field, file_path in FIELD_FILES.items():
    with open(file_path, "r", encoding="utf-8") as f:
        fields_data[field] = json.load(f)

samples = []

for _ in range(NUM_SAMPLES):
    sample = {}
    for field, values in fields_data.items():
        sample[field] = random.choice(values)  # 필드별 랜덤 선택
    samples.append(sample)

# JSONL 저장
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"✅ 합성 처방전 {NUM_SAMPLES}개 생성 완료:", SAVE_PATH)
