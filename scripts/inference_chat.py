# inference.py
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info # 사용자 정의 유틸리티 파일 필요

# ------------------------------------------------
# 1. 모델 및 프로세서 로드 함수 (한 번만 실행)
# ------------------------------------------------
def load_qwen2_vl_model(model_name: str, device: str):
    """
    Qwen2VL 모델과 프로세서를 로드합니다.
    """
    print(f"모델 '{model_name}' 로드 중...")
    
    # GPU 사용 시 torch_dtype=torch.float16, device_map="auto" 설정
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        model.eval()
        print("모델 로드 완료!")
        
        processor = AutoProcessor.from_pretrained(model_name)
        print("프로세서 로드 완료!")
        
        return model, processor
    except Exception as e:
        print(f"모델 또는 프로세서 로드 중 오류 발생: {e}")
        return None, None

# ------------------------------------------------
# 2. 추론 함수
# ------------------------------------------------
def inference_qwen2_vl(
    model, 
    processor, 
    image_url: str, 
    question: str, 
    max_new_tokens: int = 128, 
    device: str = "cpu"
) -> str:
    """
    로드된 Qwen2VL 모델을 사용하여 주어진 이미지 URL과 질문으로 추론을 수행합니다.
    
    Args:
        model: 로드된 Qwen2VLForConditionalGeneration 모델 객체.
        processor: 로드된 AutoProcessor 객체.
        image_url (str): 추론할 이미지 파일의 경로.
        question (str): 이미지에 대한 질문 (고정 또는 가변).
        max_new_tokens (int): 생성할 최대 새 토큰 수.
        device (str): 모델이 위치한 장치 ('cuda' 또는 'cpu').
        
    Returns:
        str: 모델이 생성한 답변 텍스트.
    """
    if model is None or processor is None:
        return "오류: 모델 또는 프로세서가 로드되지 않았습니다."

    print(f"\n--- 추론 시작 (Image: {image_url}) ---")
    
    # 2. 메시지 구성
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": f"<image>\n{question}"}
            ],
        }
    ]

    # 3. processor로 입력 준비
    print("입력 텐서 준비 중...")
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # process_vision_info 유틸리티 함수 사용 (원본 스크립트와 동일)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    print("입력 텐서 준비 완료!")

    # 4. 추론
    print("모델 추론 시작...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 생성된 토큰만 추출 및 디코딩
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    print("디코딩 중...")
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    print("--- 추론 완료 ---")
    return output_text[0]

# ------------------------------------------------
# 3. 실행 예시
# ------------------------------------------------
if __name__ == "__main__":
    # 설정
    MODEL_NAME = "Rfy23/qwen2vl-ko-zh"  # 사용자 병합 모델
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 고정 질문 및 이미지 경로 (테스트용)
    IMAGE_URL = "/home/jwlee/volume/Qwen2-vl-finetune-wo/scripts/test-Img/images/00003.jpg"
    FIXED_QUESTION = "这张处方上写了什么？ 尤其是药品、服用次数等，请准确全部告诉我。" 

    # 1단계: 모델 및 프로세서 로드
    qwen_model, qwen_processor = load_qwen2_vl_model(MODEL_NAME, DEVICE)

    # 2단계: 추론 실행
    if qwen_model and qwen_processor:
        result = inference_qwen2_vl(
            model=qwen_model,
            processor=qwen_processor,
            image_url=IMAGE_URL,
            question=FIXED_QUESTION,
            max_new_tokens=256, # 예시로 토큰 수를 늘릴 수 있습니다.
            device=DEVICE
        )

        print("\n=== 최종 모델 출력 ===")
        print(result)
        print("=======================")
    else:
        print("\n추론을 실행할 수 없습니다. 모델 로드에 실패했습니다.")