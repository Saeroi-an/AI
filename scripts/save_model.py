import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel # LoRA 병합을 위해 필요

# --- 1. 변수 설정 ---
# Merge된 모델을 저장할 새로운 로컬 경로
SAVE_PATH = "/home/jwlee/volume/Qwen2-vl-finetune-wo/output/merge_model/bf16" 
# 병합 전 기본 모델의 허깅페이스 경로 또는 로컬 경로 (예시)
BASE_MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct" 
# LoRA 어댑터가 저장된 경로 (예시)
LORA_ADAPTER_PATH = "/home/jwlee/volume/Qwen2-vl-finetune-wo/output/checkpoints" 

# --- 2. 기본 모델 및 컴포넌트 로드 ---
print("✅ 기본 모델, 토크나이저, 프로세서를 로드합니다...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16, # 기본 모델을 BF16으로 로드하여 메모리 효율성을 높입니다.
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# --- 3. LoRA 어댑터 로드 및 병합 (Merge) ---
print("✅ LoRA 어댑터를 로드하고 기본 모델에 병합(Merge)합니다...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# 가중치를 완전히 병합하고 LoRA 관련 메모리를 언로드합니다.
# 이 결과로 merged_model 객체에는 최종 파인튜닝된 가중치가 포함됩니다.
merged_model = model.merge_and_unload()
del model
del base_model
torch.cuda.empty_cache() # 메모리 정리

# --- 4. 최종 모델을 BF16 정밀도로 디스크에 저장 (핵심) ---
print(f"✅ 최종 모델을 BF16 정밀도로 {SAVE_PATH}에 저장합니다...")

# merged_model.save_pretrained()를 실행하며 torch_dtype=torch.bfloat16 지정
merged_model.save_pretrained(
    SAVE_PATH,
    max_shard_size="2GB",  # 대용량 모델 파일 분할 (선택 사항)
    safe_serialization=True,
    torch_dtype=torch.bfloat16  # <--- 이 설정이 최종 파일의 정밀도를 BF16으로 만듭니다!
)

# 토크나이저와 프로세서도 함께 저장
tokenizer.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)

print("---")
print(f"✅ BF16 저장 완료! 최종 모델 크기: 약 14GB (7B 기준)")
print(f"다음 단계: {SAVE_PATH} 디렉토리의 파일을 허깅페이스에 푸시하세요.")