import json
import os

# LLaVA dataset JSON 경로
dataset_file = "synth_rx/llava_receipt_dataset_ko.json"
sorted_dataset_file = "synth_rx/sorted_llava_receipt_dataset_ko.json" 

# dataset_file = "synth_rx/llava_receipt_dataset_en.json"
# sorted_dataset_file = "synth_rx/sorted_llava_receipt_dataset_en.json" 

# 00000 ~ 00799 파일 범위
all_possible_files = {f"{i:05d}.json" for i in range(800)}

# dataset 불러오기
with open(dataset_file, "r", encoding="utf-8") as f:
    llava_dataset = json.load(f)

valid_files = set()
skip_files = set()
total_conversations = 0  # 총 conversation 수

# ID 오름차순 정렬
llava_dataset.sort(key=lambda x: x["id"])

# <image> 태그 중복 제거
for entry in llava_dataset:
    convos = entry.get("conversations", [])
    total_conversations += len(convos) // 2  # human+gpt 한 쌍 = 1 conversation

    first_image_found = False
    for msg in convos:
        if "<image>" in msg.get("value", ""):
            if not first_image_found:
                first_image_found = True
            else:
                # 맨 첫번째 이후 <image> 제거
                msg["value"] = msg["value"].replace("<image>\n", "")

    # file name 추출
    file_name = os.path.basename(entry["image"]).replace(".jpg", ".json")
    valid_files.add(file_name)

# skip files 확인
skip_files = all_possible_files - valid_files

# 정렬 후 JSON으로 저장
with open(sorted_dataset_file, "w", encoding="utf-8") as f:
    json.dump(llava_dataset, f, ensure_ascii=False, indent=2)

print(f"✅ Sorted & cleaned dataset saved: {sorted_dataset_file}")
print(f"총 entries: {len(llava_dataset)}")
print("\n=== Valid files ===")
for f in sorted(valid_files):
    print(f)

print("\n=== Skip files ===")
for f in sorted(skip_files):
    print(f)

print(f"\n✅ 총 valid files: {len(valid_files)}")
print(f"❌ 총 skipped files: {len(skip_files)}")
print(f"📝 총 conversations 수: {total_conversations}")
