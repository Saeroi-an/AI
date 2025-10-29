"""
synthdog-ko 데이터셋 전처리
"""

import os
import json

# -----------------------------
# 이미지/JSON 폴더 경로
# -----------------------------
image_dir = "data/synthdog-ko/images"
json_dir = "data/synthdog-ko/annotations_json"

output_json = "synth_rx/final_qa_korean_datasets_LLaVA.json"

# -----------------------------
# 파일 정렬
# -----------------------------
image_files = sorted(os.listdir(image_dir))
json_files = sorted(os.listdir(json_dir))

assert len(image_files) == len(json_files), "이미지와 JSON 개수가 다릅니다!"

# -----------------------------
# LLaVA 포맷 변환
# -----------------------------
llava_data = []

for idx, (img_file, json_file) in enumerate(zip(image_files, json_files)):
    # 진행 상황 출력 (디버깅)
    print(f"현재 진행되는 이미지 파일: {img_file}, text 파일: {json_file}")

    json_path = os.path.join(json_dir, json_file)

    # JSON 읽기
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read()              # 파일 전체를 문자열로 읽음
        data = json.loads(content)      # 1차 JSON 디코딩
        if isinstance(data, str):       # 만약 또 문자열이면
            data = json.loads(data)     # 2차 JSON 디코딩
    
    text_sentence = data["gt_parse"]["text_sequence"]
    # with open(os.path.join(json_dir, json_file), "r", encoding="utf-8") as f:
    #   data = json.load(f)
    # text_sentence = data["gt_parse"]["text_sequence"]

    
    
    # LLaVA 구조 생성
    entry = {
        "id": img_file.split('.')[0],  # 이미지 파일명(확장자 제외) 사용
        "image": img_file,
        "conversations": [
            {"from": "human", "value": "<image>\n请提取图像中的韩语文本?"},
            {"from": "gpt", "value": text_sentence}
        ]
    }
    llava_data.append(entry)

# -----------------------------
# JSON 저장
# -----------------------------
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(llava_data, f, ensure_ascii=False, indent=2)

print(f"LLaVA 포맷 저장 완료! {len(llava_data)}개 항목, 파일: {output_json}")
