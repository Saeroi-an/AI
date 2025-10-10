# B안: Cord 데이터셋 Lora 파인튜닝 이후
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from qwen_vl_utils import process_vision_info

MODEL_PATH = "output/qwen2vl_merged"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).cuda()

image = Image.open("data/cord_sample/images/00001.jpg")
query = "<image>\n이 문서에서 알 수 있는 정보를 모두 알려줘."

inputs = tokenizer(process_vision_info([image], query), return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=512)

print("=== LoRA 파인튜닝 후 결과 ===")
print(tokenizer.decode(out[0], skip_special_tokens=True))
