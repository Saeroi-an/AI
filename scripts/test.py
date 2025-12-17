import json
import os
import torch
import re
import difflib
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# NLTK 리소스 다운로드 (최초 1회 실행 시 필요)
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# 1. 경로 및 설정
MODEL_ID = "Rfy23/qwenvl-7B-medical-ko-zh" 
JSON_FILE_PATH = "/home/jwlee/volume/Qwen2-vl-finetune-wo/data/ko_zh_datasets_4/test_zh_ko.json"
IMAGE_BASE_DIR = "/home/jwlee/volume/Qwen2-vl-finetune-wo" 

# 2. 모델 및 프로세서 로드
print(f"🚀 모델 로딩 중: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
model.eval()

# 3. 데이터 로드 및 환경 설정
with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

all_preds = []
all_labels = []
r_scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)

# 4. 추론 루프 (Inference Loop)
print(f"🔍 추론 시작 (총 {len(test_data)}개 이미지 세트)...")

for entry in tqdm(test_data):
    # 이미지 경로 정규화
    img_rel_path = entry["image"]
    image_path = os.path.join(IMAGE_BASE_DIR, img_rel_path)
    
    if not os.path.exists(image_path):
        print(f"\n⚠️ 이미지를 찾을 수 없음: {image_path}")
        continue
        
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"\n⚠️ 이미지 로드 실패: {image_path}, 에러: {e}")
        continue

    conversations = entry["conversations"]
    history_messages = []
    
    # LLaVA 형식의 멀티턴(human-gpt 쌍) 처리
    for i in range(0, len(conversations), 2):
        human_query = conversations[i]["value"].replace("<image>\n", "")
        gpt_target = conversations[i+1]["value"]
        
        # 메시지 구성 (첫 턴에만 이미지 포함)
        if i == 0:
            content = [{"type": "image", "image": image}, {"type": "text", "text": human_query}]
        else:
            content = [{"type": "text", "text": human_query}]
            
        history_messages.append({"role": "user", "content": content})
        
        # 템플릿 적용 및 텐서 생성
        text = processor.apply_chat_template(history_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
        
        # 답변 생성
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
        
        # 결과 저장
        all_preds.append(output_text)
        all_labels.append(gpt_target)
        
        # 다음 턴을 위해 모델 응답 기록 ( Assistant 역할 )
        history_messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})

# 5. 확장된 평가지표 계산 함수
def calculate_advanced_metrics(preds, labels):
    results = {
        "keyword_acc": [], "rougeL": [], "rouge1": [], 
        "bleu": [], "char_f1": [], "meteor": []
    }
    smoothie = SmoothingFunction().method1

    for p, l in zip(preds, labels):
        # 1) Keyword Match (정보 재현율)
        target_keywords = re.findall(r'[가-힣A-Z0-9]+', l)
        k_score = sum(1 for w in target_keywords if w in p) / len(target_keywords) if target_keywords else 0
        results["keyword_acc"].append(k_score)

        # 2) ROUGE (문장 흐름 및 구조)
        rs = r_scorer.score(l, p)
        results["rougeL"].append(rs['rougeL'].fmeasure)
        results["rouge1"].append(rs['rouge1'].fmeasure)

        # 3) BLEU-4 (구문 유사도)
        p_tokens = nltk.word_tokenize(p)
        l_tokens = nltk.word_tokenize(l)
        results["bleu"].append(sentence_bleu([l_tokens], p_tokens, smoothing_function=smoothie))

        # 4) METEOR (의미론적 유사도)
        results["meteor"].append(meteor_score([l_tokens], p_tokens))

        # 5) Character-level F1 (오타 및 미세 일치)
        results["char_f1"].append(difflib.SequenceMatcher(None, p.replace(" ",""), l.replace(" ","")).ratio())

    return results

# 6. 결과 집계 및 출력
print("\n📊 평가지표 산출 중...")
metrics = calculate_advanced_metrics(all_preds, all_labels)
num_samples = len(all_preds)

print("\n" + "="*60)
print(f"🔬 Qwen2-VL 도메인 특화 종합 평가 보고서 (N={num_samples})")
print("-" * 60)
print(f"📊 [정보 추출] Keyword Accuracy:   {sum(metrics['keyword_acc'])/num_samples:.4f}")
print(f"📊 [오타 보정] Char-level F1:     {sum(metrics['char_f1'])/num_samples:.4f}")
print(f"📝 [글짓기 1] ROUGE-L (흐름):     {sum(metrics['rougeL'])/num_samples:.4f}")
print(f"📝 [글짓기 2] ROUGE-1 (단어):     {sum(metrics['rouge1'])/num_samples:.4f}")
print(f"📝 [글짓기 3] METEOR (의미):      {sum(metrics['meteor'])/num_samples:.4f}")
print(f"📝 [글짓기 4] BLEU-4 (정교함):    {sum(metrics['bleu'])/num_samples:.4f}")
print(f"❌ [완전 일치] Exact Accuracy:    {accuracy_score(all_labels, all_preds):.4f}")
print("="*60)

# 분석용 샘플 1개 출력
if all_preds:
    print("\n[테스트 샘플 확인]")
    print(f"정답: {all_labels[-1]}")
    print(f"모델: {all_preds[-1]}")