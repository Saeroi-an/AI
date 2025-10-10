# import json
# import glob

# json_files = glob.glob("data/cord_sample/annotations/*.json")

# for path in json_files:
#     with open(path, "r", encoding="utf-8") as f:
#         raw = f.read()  # 문자열로 읽음

#     # 문자열 안 JSON을 dict로 변환 (2중 JSON 처리)
#     data = json.loads(raw)       # 1차 변환 → 아직 문자열일 수 있음
#     if isinstance(data, str):
#         data = json.loads(data)  # 2차 변환 → 이제 dict가 됨

#     # 확인
#     print(f"File: {path}")
#     print("Type:", type(data))  # <class 'dict'>
#     print("Top-level keys:", data.keys())
#     print("-" * 50)

import json
import glob
import os

# JSON 파일 위치
json_files = glob.glob("data/cord_sample/annotations_json/*.json")

for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id = os.path.splitext(os.path.basename(json_path))[0]
    gt_parse = data.get("gt_parse", {})

    print(f"\n===== {image_id} =====")

    menu_list = gt_parse.get("menu", [])
    print("menu 리스트 길이:", len(menu_list))

    for idx, item in enumerate(menu_list):
        print(f"\nitem {idx} 타입:", type(item))
        print("item 내용:", item)

        # 만약 dict이면 keys 확인
        if isinstance(item, dict):
            print("keys:", list(item.keys()))
        else:
            print("dict 아님, 값 그대로 출력:", item)
