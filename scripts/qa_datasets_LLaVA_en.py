import json
import random
import os
from glob import glob

# QA 템플릿
qa_templates = {
    "direct": {
        0: "What is the price of {}?",
        1: "How many {} are there?",
        2: "What is the name of the menu item with price {} and quantity {}?",
        3: "What are the price and quantity of {}?"
    },
    "structure": [
        "List {} as (quantity, item name, price).",
        "Summarize the line for {}.",
        "Show {} in the order of quantity, item name, and price."
    ],
    "reasoning": [
        "What is the most expensive item?",
        "What is the total amount?",
        "Which item has the highest quantity?"
    ]
}

# 직접 질의형을 위한 placeholder
placeholder_map = {
    0: ("nm",),                # 가격 묻는 질문 -> 메뉴 이름 사용
    1: ("nm",),                # 수량 묻는 질문 -> 메뉴 이름 사용
    2: ("price", "cnt"),       # 가격+수량으로 메뉴 이름 찾기
    3: ("nm",)                 # 가격+수량 묻는 질문 -> 메뉴 이름 사용
}


# 5~15 turn 랜덤 선택
def get_turn_count():
    return random.randint(5, 15)

# QA 유형 비율 자동 배분 (3가지 유형 모두 포함)
def allocate_qa_types(turn_count):
    min_each = 1
    remaining = turn_count - 3 * min_each
    direct = min_each + random.randint(0, remaining)
    remaining2 = remaining - (direct - min_each)
    structure = min_each + random.randint(0, remaining2)
    reasoning = turn_count - direct - structure
    return {"direct": direct, "structure": structure, "reasoning": reasoning}

# 단일 conversation 생성
def generate_conversation(receipt_json, used_items_global=set()):
    items = receipt_json["gt_parse"]["menu"] # items는 리스트, 각 요소는 dict
    turn_count = get_turn_count()
    qa_allocation = allocate_qa_types(turn_count)
    conversation = []

    # 이미 사용한 항목 제외
    available_items = [i for i in items if i["nm"] not in used_items_global]
    if not available_items:
        return None, used_items_global

    # 샘플링
    num_items_needed = min(len(available_items), turn_count)
    sampled_items = random.sample(available_items, num_items_needed) # sampled_items는 list of dict
    used_items_local = set() # 형태: {요소1, 요소2, ...} 같은 conversation 내 중복 질문 방지

    # 직접질의형
    for _ in range(qa_allocation["direct"]):
        if not sampled_items:
            break
        item = sampled_items.pop(0)
        
        #중복 방지
        used_items_local.add(item["nm"]) 
        used_items_global.add(item["nm"]) 
        
        template_id, question_template = random.choice(list(qa_templates["direct"].items()))
        placeholders = placeholder_map[template_id]
        
        
        # placeholder_values = tuple(item[p] for p in placeholders)
        # question = f"<image>\n" + question_template.format(placeholder_value)

        # placeholder 값 준비
        if len(placeholders) == 1:
            placeholder_value = item[placeholders[0]]
        else:
            placeholder_value = tuple(item[p] for p in placeholders)

        # 질문 생성
        question = question_template.format(*placeholder_value) if isinstance(placeholder_value, tuple) else question_template.format(placeholder_value)

        # answer 매핑
        if template_id == 0:           
            answer = item["price"]
        elif template_id == 1:         
            answer = item["cnt"]
        elif template_id == 2:         
            answer = item["nm"]
        elif template_id == 3:         
            answer = f"price: {item['price']}, count: {item['cnt']}"

        
        conversation.append({"from": "human", "value": question})
        conversation.append({"from": "gpt", "value": answer})

    # 구조형
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

    # 논리 추론형
    reasoning_questions = [
        ("Which menu item is the most expensive?", max(items, key=lambda x: price_to_int(x["price"]))["nm"]),
        ("What is the total sum?", str(sum(price_to_int(x["price"]) for x in items))),
        ("Which item has the highest quantity?", max(items, key=lambda x: cnt_to_int(x["cnt"]))["nm"])
    ]

    for _ in range(qa_allocation["reasoning"]):
        question, answer = random.choice(reasoning_questions)
        conversation.append({"from": "human", "value": f"{question}"})
        conversation.append({"from": "gpt", "value": answer})

    return conversation, used_items_global

def price_to_int(price_str):
    # 콤마 제거
    price_str = price_str.replace(",", "")
    # 점 제거 (천 단위 구분일 경우)
    price_str = price_str.replace(".", "")
    try:
        return int(price_str)
    except ValueError:
        return 0  # 혹은 적절한 기본값

def cnt_to_int(cnt_str):
    try:
        # 숫자만 추출
        return int(''.join(filter(str.isdigit, str(cnt_str))))
    except ValueError:
        return 0


def add_image_tag_to_conversation(conversations, description="The following image is a receipt. Please answer questions about it. "):
    for msg in conversations:
        if msg["from"] == "human":
            msg["value"] = f"<image>\n{description}\n" + msg["value"]
            break
    return conversations

# 폴더 내 모든 JSON 처리
def create_llava_dataset_from_folder(json_folder, output_file, min_convo=1, max_convo=3):
    json_files = glob(os.path.join(json_folder, "*.json"))
    llava_dataset = []
    skipped_files_not_dict = []
    skipped_files_included_sub = []
    

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            receipt_json = json.load(f)

        image_filename = receipt_json.get("image", os.path.basename(json_file).replace(".json", ".jpg"))
        
        items = receipt_json["gt_parse"]["menu"]
        valid_items = []

        for idx, item in enumerate(items):
            # 1. dict 여부 체크
            if not isinstance(item, dict):
                print(f"💚딕셔너리가 아니라 skip된 파일: {os.path.basename(json_file)} menu index {idx}: 딕셔너리가 아니라 str임. -> {item}")
                skipped_files_not_dict.append(os.path.basename(json_file))
                continue

            # 2. price / unitprice 처리
            if "unitprice" in item:
                
                if "price" not in item:
                    item["price"] = item["unitprice"]

                
                elif item["price"] != item["unitprice"]:
                    print(f"⚠️ price와 unitprice 값 불일치: {os.path.basename(json_file)} idx {idx} -> price={item['price']}, unitprice={item['unitprice']}")
                    
                    item["price"] = item["unitprice"]

                
                item.pop("unitprice", None)

            
            # 3. cnt 기본값 설정
            if "cnt" not in item:
                item["cnt"] = "1"
            
            # 4. sub_keys 존재 여부 체크
            sub_keys = [k for k in item.keys() if k not in ["nm", "cnt", "price"]]
            if sub_keys:
                print(f"❤️sub_key가 있어서 skip된 파일: {os.path.basename(json_file)} due to unexpected sub keys in menu index {idx}: {sub_keys}")
                skipped_files_included_sub.append(os.path.basename(json_file))
                continue

            valid_items.append(item)
                
        num_convos = random.randint(min_convo, max_convo)
        used_items_global = set() # 이미지 전체에서 이미 사용된 항목 추적 -> 같은 이미지 내에서 이미 질문한 항목명을 저장

        for aug_index in range(num_convos):
            
            print(f"처리중인 파일명: {os.path.basename(json_file)}, aug_index: {aug_index}") 
            
            # conversation, used_items_global = generate_conversation(receipt_json, used_items_global)
            # valid_items만 generate_conversation에 넘김
            conversation, used_items_global = generate_conversation(
                {"gt_parse": {"menu": valid_items}},
                used_items_global
            )
            
            if conversation is None:
                continue
            
             # 후처리로 첫 번째 질문에만 <image> 태그 추가
            conversation = add_image_tag_to_conversation(conversation)
            
            entry = {
                "id": f"{receipt_json.get('id', os.path.splitext(os.path.basename(json_file))[0])}_aug{aug_index}",
                "image": image_filename,
                "conversations": conversation
            }
            llava_dataset.append(entry)

    # 전체 리스트를 한 번에 JSON으로 저장
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(llava_dataset, f_out, ensure_ascii=False, indent=2)

    print(f"LLaVA dataset generated 저장!!❤️: {output_file}, total entries: {len(llava_dataset)}")
    if skipped_files_not_dict:
        print("========================={💚dict가 아니라서 skip된 파일들💚}==========================}")
        skipped_files_not_dict = list(set(skipped_files_not_dict))  # 중복 제거
        print("Skipped files due to invalid menu items:", skipped_files_not_dict)

    if skipped_files_included_sub:
        print("========================={❤️sub class가 있어서 skip된 파일들❤️}==========================}")
        skipped_files_included_sub = list(set(skipped_files_included_sub))  # 중복 제거
        print("Skipped files due to invalid menu items:", skipped_files_included_sub)


if __name__ == "__main__":
    json_folder = "data/cord_sample/annotations_json" 
    output_file = "synth_rx/llava_receipt_dataset_en.json"
    create_llava_dataset_from_folder(json_folder, output_file)
