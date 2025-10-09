import json
import glob

json_files = glob.glob("data/cord_sample/annotations/*.json")

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()  # 문자열로 읽음

    # 문자열 안 JSON을 dict로 변환 (2중 JSON 처리)
    data = json.loads(raw)       # 1차 변환 → 아직 문자열일 수 있음
    if isinstance(data, str):
        data = json.loads(data)  # 2차 변환 → 이제 dict가 됨

    # 확인
    print(f"File: {path}")
    print("Type:", type(data))  # <class 'dict'>
    print("Top-level keys:", data.keys())
    print("-" * 50)
