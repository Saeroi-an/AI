import json
import re
from itertools import chain

# 파일 경로
dict_file = "synth_rx/llava_receipt_dataset_dict.json"
en_file = "synth_rx/llava_receipt_dataset_en.json"
output_file = "synth_rx/qa_datasets_LLaVA_for_finetune.json"

# flatten loader
def load_json_flat(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data[0], list):
        flat_data = list(chain.from_iterable(data))
        return flat_data
    return data

# JSON 로드 및 flatten
dict_data = load_json_flat(dict_file)
en_data = load_json_flat(en_file)

print(f"✅ Loaded dict_data: {len(dict_data)} entries")
print(f"✅ Loaded en_data: {len(en_data)} entries")

# 각 파일에서 원본 ID 리스트 추출 (증강 제외)
id_pattern = re.compile(r"(\d{5})")  # 00000~99999 기준
datasets_dict_list = []
datasets_default_list = []

for entry in dict_data:
    match = id_pattern.match(entry["id"])
    if match:
        datasets_dict_list.append(match.group(1))

for entry in en_data:
    match = id_pattern.match(entry["id"])
    if match:
        datasets_default_list.append(match.group(1))

print(f"Dict file original IDs: {len(set(datasets_dict_list))}")
print(f"En file original IDs: {len(set(datasets_default_list))}")

# 병합
merged_data = dict_data + en_data

# ID 기준 오름차순 정렬 (증강 포함)
def id_sort_key(entry):
    base_id_match = id_pattern.match(entry["id"])
    base_id = int(base_id_match.group(1)) if base_id_match else 99999
    aug_match = re.search(r"_aug(\d+)$", entry["id"])
    aug_num = int(aug_match.group(1)) if aug_match else -1
    return (base_id, aug_num)

merged_data.sort(key=id_sort_key)

# 결과 저장
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(merged_data, f_out, ensure_ascii=False, indent=2)

# 최종 통계
all_ids = set(range(0, 800))
present_ids = set(int(entry_id) for entry_id in datasets_dict_list + datasets_default_list)
missing_ids = sorted(list(all_ids - present_ids))

print(f"✅ Merged dataset saved to {output_file}!")
print(f"Total merged entries: {len(merged_data)}")
print(f"Missing original IDs (00000~00799): {missing_ids}")
print(f"Total unique original IDs in merged dataset: {len(present_ids)}")
