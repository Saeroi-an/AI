import os
import getpass
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")
    
    
HF_REPO_ID = "skt/ko-gpt-trinity-1.2B-v0.5"
MODEL_NAME = "Rfy23/qwen2vl-ko-zh"
IMAGE_URL = "/home/jwlee/volume/Qwen2-vl-finetune-wo/scripts/test-Img/images/00003.jpg"
FIXED_QUESTION = "这张处方上写了什么？ 尤其是药品、服用次数等，请准确全部告诉我。" 
tools = [vqa_model]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_MODEL = None
QWEN_PROCESSOR = None

# 3. 모델/프로세서 로드 함수 (스크립트 시작 시 단 한 번 호출)
def load_qwen_components(model_name: str, device: str):
    """Qwen2VL 모델과 프로세서를 전역으로 로드"""
    global QWEN_MODEL, QWEN_PROCESSOR, DEVICE
    
    if QWEN_MODEL is not None:
        logger.info("모델이 이미 로드되었습니다.")
        return

    logger.info(f"🚀 VQA 모델 '{model_name}' 전역 로드 시작 (Device: {device})...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None

    try:
        print("모델 로드 중...")
        # ⚠️ Qwen2VLForConditionalGeneration 클래스 경로는 사용 환경에 맞게 조정 필요
        QWEN_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        QWEN_MODEL.eval()
        print("모델 로드 완료!")
        QWEN_PROCESSOR = AutoProcessor.from_pretrained(model_name)
        logger.info("✅ 모델 및 프로세서 전역 로드 완료!")
        
    except Exception as e:
        logger.error(f"❌ VQA 모델 로드 중 오류 발생: {e}")
        raise

# 4. VQA 툴 정의 (전역 모델 사용 및 인자 활용)
# ⚠️ process_vision_info 유틸리티 함수가 전역으로 정의되어 있어야 함
@tool
def vqa_model(image_path: str, question: str = FIXED_QUESTION, max_new_tokens: int = 128) -> str:
    """
    이미지 파일 경로와 텍스트 질문을 입력으로 받아, Qwen2VL 모델을 사용하여
    해당 이미지에 대한 질문에 답변을 생성합니다. (Visual Question Answering)
    """
    global QWEN_MODEL, QWEN_PROCESSOR, DEVICE
    
    if QWEN_MODEL is None or QWEN_PROCESSOR is None:
        return "오류: Qwen2VL 모델이 전역으로 초기화되지 않았습니다. load_qwen_components를 먼저 호출하세요."
        
    try:
        # 2. 메시지 구성 (함수 인자를 사용하도록 수정)
        messages = [
            {
                "role": "user",
                "content": [
                    # ⚠️ type: "image", image: image_path 포맷은 Qwen2VL에 맞게 조정 필요
                    {"type": "image", "image": image_path}, 
                    {"type": "text", "text": f"<image>\n{question}"}
                ],
            }
        ]
        
        logger.info(f"🖼️ VQA 추론 시작 (Image: {image_path}) - Question: {question[:50]}...")
        
        # 3. processor로 입력 준비 (원본 스크립트 로직 유지)
        print("입력 텐서 준비 중...")
        text_input = QWEN_PROCESSOR.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # ⚠️ process_vision_info 함수는 Qwen2VL 사용 환경에 정의되어야 함
        image_inputs, _ = process_vision_info(messages) 

        inputs = QWEN_PROCESSOR(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        # 4. 추론
        with torch.no_grad():
            generated_ids = QWEN_MODEL.generate(**inputs, max_new_tokens=max_new_tokens)

        # 생성된 토큰 추출 및 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = QWEN_PROCESSOR.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        logger.info("✅ VQA 추론 완료.")
        return output_text[0]

    except FileNotFoundError:
        return f"오류: 지정된 이미지 경로 '{image_path}'를 찾을 수 없습니다."
    except Exception as e:
        return f"VQA 처리 중 예상치 못한 오류 발생: {str(e)}"


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized assistant that uses the vqa_model tool for visual questions."), 
    ("human", "{input}"), 
    ("placeholder", "{agent_scratchpad}"),
])

llm = HuggingFacePipeline.from_model_id( # ChatHuggingFace
    model_id=HF_REPO_ID,
    task="text-generation",
    model_kwargs={"temperature": 0.1, "max_length": 512}
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": f"이미지 {IMAGE_URL}에 무엇이 쓰여 있는지 확인해 줘." })