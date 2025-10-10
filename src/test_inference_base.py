    
# Qwen2-vl 모델 파인튜닝 안했을 때 추론
    


from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from qwen_vl_utils import process_vision_info

# 사전학습된 모델 로드 (2B 모델)
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()

# 테스트 이미지
image = Image.open("data/cord_sample/images/00001.jpg")
query = "<image>\n이 문서에서 알 수 있는 정보를 모두 알려줘."

inputs = tokenizer(process_vision_info([image], query), return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=512)

print("=== LoRA 파인튜닝 전 결과 ===")
print(tokenizer.decode(out[0], skip_special_tokens=True))
