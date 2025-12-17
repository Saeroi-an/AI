# Rfy23/qwenvl-7B-medical-ko-zh

**본 프로젝트는 국내 의료 서비스를 이용하는 외국인을 위한 의료 문서 인식 AI 모델입니다. Vision Language Model(VLM)을 기반으로 한국의 처방전, 건강검진 결과지, 진료비 영수증 등 복잡한 의료 문서를 정밀하게 분석하고 인식합니다.**

<img src="image/main_ai.png">

## 🌐 Model Fine-tuning
base model: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
🤗 fine-tuning model: [qwenvl-7B-medical-ko-zh](https://huggingface.co/Rfy23/qwenvl-7B-medical-ko-zh)

효율적인 자원 사용과 정확한 OCR 성능을 위해 Hybrid Fine-tuning 전략을 채택했습니다.

- Vision Tower & Merger (Full Fine-tuning): 처방전의 미세한 한글 획(ㅗ, ㅜ 등) 인식 성능을 높이기 위해 언프리징(Unfrozen)하여 직접 학습.
- LLM (LoRA): 모델 본체 가중치는 동결(Frozen)하고, 핵심 어텐션 레이어(q_proj, v_proj)에 LoRA 어댑터를 적용하여 메모리 효율성 확보 및 기존 언어 지식 보존.
---
## 🌐 Files Structure

```text
.
├── data/                # 합성 처방전 및 영수증 데이터셋 (.json 및 이미지 파일)
├── scripts/             # 모델 학습 및 평가를 위한 Shell 스크립트
│   └── finetune_lora.sh
├── src/                 # 메인 소스 코드
│   ├── dataset/         # 데이터 로딩 및 전처리 로직 (VLM 형식 변환)
│   ├── loss/            # 학습을 위한 커스텀 손실 함수 정의
│   ├── model/           # Qwen2-VL 모델 아키텍처 및 설정 관련 코드
│   ├── serve/           # 추론(Inference) 및 API 서빙 관련 코드
│   ├── train/           # SFT(Supervised Fine-Tuning) 메인 실행 스크립트
│   └── trainer/         # 파이토치/DeepSpeed 기반 학습 엔진 관리
└── output/              # 체크포인트 및 학습 로그 저장 폴더 ➡️ .gitignore
```
---

## 🌐 Dataset Info
<img src="image/datasetsInfo.png">

- Train Data (3,636 samples): 합성된 한국어 처방전(90%) + Key-Value 학습용 영수증(10)
- Test Data (481 samples): 실제 처방전 양식 기반 테스트셋

데이터셋은 [여기](https://github.com/Saeroi-an/AI/tree/main/data)에서 찾을 수 있습니다. 추후에 허깅페이스에 업로드 할 예정입니다.



---

## 🌐 How to Train
**requirements.txt 설치**
```bash
pip install -r requirements.txt
```

**lora fine-tuning 실행**
```bash
cd scripts
bash finetune_lora.sh
```

**주요 파라미터 설정**
- Precision: bf16
- Optimization: DeepSpeed ZeRO-3 Offload, Liger Kernel
- Learning Rate: LLM($5\times10^{-6}$), Vision($2\times10^{-6}$), Merger($1\times10^{-5}$)
- Batch Size: 32 (Global) / Epochs: 5

  
