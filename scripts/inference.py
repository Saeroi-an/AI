import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ------------------------------
# 1. 모델 & 프로세서 로드
# ------------------------------
print("모델 로드 중...")
model_name = "Rfy23/qwen2vl-ko-zh"  # 내 병합 모델
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    device_map="auto" if device=="cuda" else None
)
model.eval()
print("모델 로드 완료!")

processor = AutoProcessor.from_pretrained(model_name)
print("프로세서 로드 완료!")

# ------------------------------
# 2. 고정 질문 + 이미지 URL
# ------------------------------
image_url = "/home/jwlee/volume/Qwen2-vl-finetune-wo/scripts/test-out/images/00001.jpg"
fixed_question = "这张处方上写了什么？"  # 원하는 고정 질문

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": f"<image>\n{fixed_question}"}
        ],
    }
]

# ------------------------------
# 3. processor로 입력 준비
# ------------------------------
print("입력 텐서 준비 중...")
text_input = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, _ = process_vision_info(messages)

inputs = processor(
    text=[text_input],
    images=image_inputs,
    # videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(device)
print("입력 텐서 준비 완료!")

# ------------------------------
# 4. 추론
# ------------------------------
print("모델 추론 시작...")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

print("디코딩 중...")
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("모델 출력:")
print(output_text[0])
