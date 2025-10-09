import json
import glob
import os

# JSON 파일 위치
json_files = glob.glob("data/cord_sample/annotations/*.json")
# 저장할 JSONL 파일 위치
save_path = "data/cord_sample_qwen_full.jsonl"

# 이미지 파일 위치
image_files = glob.glob("data/cord_sample/images/*.jpg")
# image_id와 파일 경로 매핑
image_dict = {int(os.path.splitext(os.path.basename(f))[0]): f for f in image_files}

samples = []

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    
    # 1차 JSON 디코딩
    data = json.loads(raw)
    
    # 2차 JSON 디코딩 (이중 문자열)
    if isinstance(data, str):
        data = json.loads(data)
    
    image_id = data.get("meta", {}).get("image_id")
    image_path = image_dict.get(image_id)
    
    if image_path is None:
        print(f"⚠️ 이미지 {image_id} 없음, 스킵합니다.")
        continue

    # conversations 생성
    conv = []

    # gt_parse 처리
    gt_parse = data.get("gt_parse", {})

    # menu
    for item in gt_parse.get("menu", []):
        if isinstance(item, dict):
            for k, v in item.items():
                conv.append({"from": "human", "value": f"이 항목의 {k}를 알려줘"})
                conv.append({"from": "gpt", "value": v})
        else:
            conv.append({"from": "human", "value": "menu 항목을 알려줘"})
            conv.append({"from": "gpt", "value": str(item)})

    # sub_total
    sub_total = gt_parse.get("sub_total", {})
    if isinstance(sub_total, dict):
        for k, v in sub_total.items():
            conv.append({"from": "human", "value": f"이 항목의 {k}를 알려줘"})
            conv.append({"from": "gpt", "value": v})

    # total
    total = gt_parse.get("total", {})
    if isinstance(total, dict):
        for k, v in total.items():
            conv.append({"from": "human", "value": f"이 항목의 {k}를 알려줘"})
            conv.append({"from": "gpt", "value": v})

    # valid_line, roi, repeating_symbol, dontcare 포함
    for field in ["valid_line", "roi", "repeating_symbol", "dontcare"]:
        if field in data:
            conv.append({"from": "human", "value": f"{field} 정보를 보여줘"})
            conv.append({"from": "gpt", "value": json.dumps(data[field], ensure_ascii=False)})

    # meta 정보 포함
    meta = data.get("meta", {})
    for k, v in meta.items():
        conv.append({"from": "human", "value": f"메타 데이터 {k}를 알려줘"})
        conv.append({"from": "gpt", "value": json.dumps(v, ensure_ascii=False)})

    # 최종 sample
    samples.append({
        "image": image_path,
        "conversations": conv
    })

# JSONL 저장
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print("✅ 모든 필드 포함하여 Qwen2-VL JSONL 생성 완료:", save_path)
