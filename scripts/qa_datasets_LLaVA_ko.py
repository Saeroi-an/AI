import json
import random
import os
from glob import glob
from datetime import datetime

# === 로그 파일 설정 ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"skip_log_{timestamp}.txt")

def log_to_file(message: str):
    """모든 로그를 파일에도 저장"""
    with open(log_file_path, "a", encoding="utf-8") as log_f:
        log_f.write(message + "\n")


# === QA 템플릿 ===
qa_templates = {
     "direct": {
        0: "{}의 가격은 얼마야?",
        1: "{}의 수량은?",
        2: "가격이 {}이고 수량이 {}인 메뉴 이름은?",
        3: "{} 메뉴의 가격과 수량은?"
    },
    "structure": [
        "{} 항목을 (수량, 품목명, 가격)으로 정리해줘.",
        "{} 줄을 정리해서 보여줘.",
        "{} 수량/품목/가격 순으로 알려줘."
    ],
    "reasoning": [
        "가장 비싼 메뉴는 뭐야?",
        "총합은 얼마야?",
        "가장 많은 수량의 항목은?"
    ]
}

placeholder_map = {
    0: ("nm",),
    1: ("nm",),
    2: ("price", "cnt"),
    3: ("nm",)
}


def get_turn_count():
    return random.randint(5, 15)


def allocate_qa_types(turn_count):
    min_each = 1
    remaining = turn_count - 3 * min_each
    direct = min_each + random.randint(0, remaining)
    remaining2 = remaining - (direct - min_each)
    structure = min_each + random.randint(0, remaining2)
    reasoning = turn_count - direct - structure
    return {"direct": direct, "structure": structure, "reasoning": reasoning}


def price_to_int(price_str):
    price_str = price_str.replace(",", "").replace(".", "")
    try:
        return int(price_str)
    except ValueError:
        return 0


def cnt_to_int(cnt_str):
    try:
        return int(''.join(filter(str.isdigit, str(cnt_str))))
    except ValueError:
        return 0


def add_image_tag_to_conversation(conversations, description="영수증이야. 너는 이거에 대해 질문을 답해줘"):
    for msg in conversations:
        if msg["from"] == "human":
            msg["value"] = f"<image>\n{description}\n" + msg["value"]
            break
    return conversations


def generate_conversation(receipt_json, used_items_global=set()):
    items = receipt_json["gt_parse"]["menu"]
    turn_count = get_turn_count()
    qa_allocation = allocate_qa_types(turn_count)
    conversation = []

    available_items = [i for i in items if i["nm"] not in used_items_global]
    if not available_items:
        return None, used_items_global

    num_items_needed = min(len(available_items), turn_count)
    sampled_items = random.sample(available_items, num_items_needed)
    used_items_local = set()

    for _ in range(qa_allocation["direct"]):
        if not sampled_items:
            break
        item = sampled_items.pop(0)
        used_items_local.add(item["nm"])
        used_items_global.add(item["nm"])

        template_id, question_template = random.choice(list(qa_templates["direct"].items()))
        placeholders = placeholder_map[template_id]

        if len(placeholders) == 1:
            placeholder_value = item[placeholders[0]]
        else:
            placeholder_value = tuple(item[p] for p in placeholders)

        question = question_template.format(*placeholder_value) if isinstance(placeholder_value, tuple) else question_template.format(placeholder_value)

        if template_id == 0:
            answer = item["price"]
        elif template_id == 1:
            answer = item["cnt"]
        elif template_id == 2:
            answer = item["nm"]
        elif template_id == 3:
            answer = f"price: {item['price']}, cnt: {item['cnt']}"

        conversation.append({"from": "human", "value": question})
        conversation.append({"from": "gpt", "value": answer})

    for _ in range(qa_allocation["structure"]):
        remaining_items = [i for i in items if i["nm"] not in used_items_local]
        if not remaining_items:
            break
        item = remaining_items.pop(0)
        used_items_local.add(item["nm"])
        used_items_global.add(item["nm"])
        question = random.choice(qa_templates["structure"]).format(item["nm"])
        answer = f"({item['cnt']}, {item['nm']}, {item['price']})"
        conversation.append({"from": "human", "value": question})
        conversation.append({"from": "gpt", "value": answer})

    reasoning_questions = [
        ("가장 비싼 메뉴는 뭐야?", max(items, key=lambda x: price_to_int(x["price"]))["nm"]),
        ("총합은 얼마야?", str(sum(price_to_int(x["price"]) for x in items))),
        ("가장 많은 수량의 항목은?", max(items, key=lambda x: cnt_to_int(x["cnt"]))["nm"])
    ]

    for _ in range(qa_allocation["reasoning"]):
        question, answer = random.choice(reasoning_questions)
        conversation.append({"from": "human", "value": f"<image>\n{question}"})
        conversation.append({"from": "gpt", "value": answer})

    return conversation, used_items_global


# === 메인 ===
def create_llava_dataset_from_folder(json_folder, output_file, min_convo=1, max_convo=3):
    json_files = glob(os.path.join(json_folder, "*.json"))
    llava_dataset = []
    skipped_files = []

    print(f"📁 Processing folder: {json_folder}")
    print(f"🧾 Found {len(json_files)} JSON files.")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                receipt_json = json.load(f)
        except Exception as e:
            msg = f"❌ JSON load error in {json_file}: {e}"
            print(msg)
            log_to_file(msg)
            skipped_files.append(os.path.basename(json_file))
            continue

        image_filename = receipt_json.get("image", os.path.basename(json_file).replace(".json", ".jpg"))
        items = receipt_json.get("gt_parse", {}).get("menu", [])
        valid_items = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                msg = f"💚 딕셔너리가 아니라 skip된 파일: {os.path.basename(json_file)} menu index {idx}: str임 -> {item}"
                log_to_file(msg)
                skipped_files.append(os.path.basename(json_file))
                continue

            if "unitprice" in item and "price" not in item:
                item["price"] = item.pop("unitprice")

            if "cnt" not in item:
                item["cnt"] = "1"

            sub_keys = [k for k in item.keys() if k not in ["nm", "cnt", "price"]]
            if sub_keys:
                msg = f"❤️ sub_key로 skip: {os.path.basename(json_file)} idx {idx}: {sub_keys}"
                log_to_file(msg)
                skipped_files.append(os.path.basename(json_file))
                continue

            valid_items.append(item)

        num_convos = random.randint(min_convo, max_convo)
        used_items_global = set()

        for aug_index in range(num_convos):
            conversation, used_items_global = generate_conversation({"gt_parse": {"menu": valid_items}}, used_items_global)
            if conversation is None:
                continue

            conversation = add_image_tag_to_conversation(conversation)

            entry = {
                "id": f"{receipt_json.get('id', os.path.splitext(os.path.basename(json_file))[0])}_aug{aug_index}",
                "image": image_filename,
                "conversations": conversation
            }
            llava_dataset.append(entry)

    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(llava_dataset, f_out, ensure_ascii=False, indent=2)

    print(f"✅ LLaVA dataset saved: {output_file}")
    print(f"💾 Total entries: {len(llava_dataset)}")
    print(f"🧾 Total skipped files: {len(set(skipped_files))}")
    print(f"📄 Full log saved to: {log_file_path}")




if __name__ == "__main__":
    json_folder = "data/cord_sample/annotations_json" 
    output_file = "synth_rx/llava_receipt_dataset_ko.json"
    create_llava_dataset_from_folder(json_folder, output_file)
