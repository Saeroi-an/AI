from transformers import AutoTokenizer
from peft import PeftModel
from PIL import Image
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model = PeftModel.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    "output/lora_merged"
).to("cuda")

image = Image.open("data/synth_rx/rx_0.png").convert("RGB")
prompt = "환자 이름과 약 이름을 JSON으로 정리해 중국어로 번역해줘"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
