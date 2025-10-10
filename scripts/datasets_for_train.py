import json
import glob
import os

# JSON 파일 위치
json_files = glob.glob("data/cord_sample/annotations_json/*.json")
# 저장할 JSON 파일 위치
save_path = "synth_rx/train_cord.json"

# 이미지 파일 위치
image_files = glob.glob("data/cord_sample/images/*.jpg")
# 파일명과 경로 매핑
image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}

samples = []

for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id = os.path.splitext(os.path.basename(json_path))[0]
    image_path = os.path.basename(image_dict.get(image_id, image_id + ".jpg"))

    gt_parse = data.get("gt_parse", {})
    conv = []

    # ---- 메뉴 항목 처리 ----
    for item in gt_parse.get("menu", []):
        if not isinstance(item, dict):
            continue  # dict 아닌 경우 스킵

        for k in ["nm", "cnt", "price", "unitprice"]:  # 일부 데이터에는 unitprice 존재
            if k not in item:
                continue

            quad = None
            for line in data.get("valid_line", []):
                if line.get("category") != f"menu.{k}":
                    continue
                for word in line.get("words", []):
                    # 텍스트 매칭 (공백 제거)
                    if isinstance(item[k], str) and word.get("text", "").replace(" ", "") == item[k].replace(" ", ""):
                        quad = word["quad"]
                        break
                if quad:
                    break

            if quad:
                conv.append({
                    "from": "human",
                    "value": f"이 항목의 {k}를 알려줘. 정답은 제공한 좌표 근처에 있어. (quad: {json.dumps(quad, ensure_ascii=False)})"
                })
                conv.append({
                    "from": "gpt",
                    "value": item[k]
                })

    # ---- total 처리 ----
    total = gt_parse.get("total", {})
    for k, v in total.items():
        quad = None
        for line in data.get("valid_line", []):
            if line.get("category") == f"total.{k}" and len(line.get("words", [])) > 0:
                quad = line["words"][0]["quad"]
                break
        if quad:
            conv.append({
                "from": "human",
                "value": f"제공한 좌표를 보고 이 항목의 {k}를 알려줘 (quad: {json.dumps(quad, ensure_ascii=False)})"
            })
            conv.append({
                "from": "gpt",
                "value": v
            })

    # ---- 최종 sample 추가 ----
    samples.append({
        "id": image_id,
        "image": image_path,
        "conversations": conv
    })

# JSON 저장
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print("✅ Qwen2-VL JSON 생성 완료:", save_path)
