import json
import random
import os
from glob import glob

# ===== QA 템플릿 =====
qa_templates = {
    "patient_info": "请告诉我这张处方上的病人姓名和身份证号码。",
    "disease": [
        "我得了什么病？",
        "我得的是什么病？",
        "医生给我诊断的是什么？",
        "看了处方，你觉得我可能得了什么病？",
        "这是我的处方，我得了什么病？"
    ],
    "medication": [
        "我拿到了药方，这是什么药？",
        "告诉我处方上的药物是什么。",
        "我吃的是什么药？",
        "我应该怎么吃这些药？",
        "药店给我了什么药？",
        "我在药店拿了药，这是什么药，要怎么吃，要吃多少？",
        "这是什么药？怎么吃，吃多少？",
        "请把处方上的药都解释一下。"
    ],
    "summary": [
        "请综合说明这张处方。",
        "这张处方上写了什么？"
    ]
}

# ===== 첫 번째 human 메시지에만 이미지 태그 추가 =====
def add_image_tag_to_conversation(conversations, description="以下是韩语处方"):
    for msg in conversations:
        if msg["from"] == "human":
            msg["value"] = f"<image>\n{description}\n" + msg["value"]
            break
    return conversations

# ===== 단일 conversation 생성 =====
def generate_conversation(data):
    conversations = []

    # 1️⃣ 患者信息
    patient_q = qa_templates["patient_info"]
    patient_a = f"病人姓名: {data.get('name','')}, 身份证号: {data.get('dob','')}"
    conversations.append({"from": "human", "value": patient_q})
    conversations.append({"from": "gpt", "value": patient_a})

    # 2️⃣ 疾病诊断 (질문만 랜덤)
    disease_q = random.choice(qa_templates["disease"])
    codes = [data.get(f"code_{i}") for i in ['A','B','C','D'] if data.get(f"code_{i}")]
    disease_a = f"疾病编码: {', '.join(codes)}。"
    conversations.append({"from": "human", "value": disease_q})
    conversations.append({"from": "gpt", "value": disease_a})

    # 3️⃣ 药物信息 (질문만 랜덤)
    med_q = random.choice(qa_templates["medication"])
    meds_list = []
    for i in ['A','B','C','D']:
        med = data.get(f"medication_{i}")
        p1 = data.get(f"period_{i}_1")
        p2 = data.get(f"period_{i}_2")
        p3 = data.get(f"period_{i}_3")
        if med:
            meds_list.append(f"处方药物: {med}, 每次剂量: {p1}, 每日服用次数: {p2}, 总疗程: {p3}")
    med_a = "\n".join(meds_list)
    conversations.append({"from": "human", "value": med_q})
    conversations.append({"from": "gpt", "value": med_a})

    # 4️⃣ 处方总结 (질문만 랜덤)
    summary_q = random.choice(qa_templates["summary"])
    summary_lines = [
        f"病人姓名: {data.get('name','')}",
        f"身份证号: {data.get('dob','')}",
        f"处方日期: {data.get('date','')}"
    ]
    for c in ['A','B','C','D']:
        code = data.get(f"code_{c}")
        if code:
            summary_lines.append(f"疾病编码{c}: {code}")
    summary_lines += meds_list
    summary_a = "\n".join(summary_lines)
    conversations.append({"from": "human", "value": summary_q})
    conversations.append({"from": "gpt", "value": summary_a})

    # 5️⃣ 첫 번째 human 메시지에 이미지 태그 추가
    conversations = add_image_tag_to_conversation(conversations)
    return conversations

# ===== 폴더 내 모든 JSON 처리 =====
def create_llava_dataset(json_folder, output_file):
    json_files = glob(os.path.join(json_folder, "*.json"))
    dataset = []

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        entry_id = data.get("id", os.path.splitext(os.path.basename(jf))[0])
        image_name = data.get("image", f"{entry_id}.jpg")

        conv = generate_conversation(data)
        dataset.append({
            "id": entry_id,
            "image": image_name,
            "conversations": conv
        })

    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(dataset, f_out, ensure_ascii=False, indent=2)
    print(f"✅ 已生成 {len(dataset)} 条数据：{output_file}")

# ===== 실행 =====
if __name__ == "__main__":
    json_folder = "data/synth_prescriptions/annotation_json"
    output_file = "synth_rx/llava_rx_dataset_prescriptions_zh.json"
    create_llava_dataset(json_folder, output_file)
