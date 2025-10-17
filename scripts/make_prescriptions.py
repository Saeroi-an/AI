# make_prescriptions.py
import json, os, random, re, subprocess
from datetime import datetime
import fitz  # PyMuPDF

# ========= 기본 설정 =========
BASE_DIR   = os.path.dirname(__file__)
FIELDS_DIR = os.path.join(BASE_DIR, "fields")

TEMPLATE_PDF = os.path.join(BASE_DIR, "Prescription_Template.pdf")

# 출력 폴더
OUT_DIR = os.path.join(BASE_DIR, "out")

# 폰트: 동일 폴더의 NotoSansKR-Regular.ttf 사용 (한글/영문 OK)
FONT_PATH    = os.path.abspath(os.path.join(BASE_DIR, "NotoSansKR-Regular.ttf"))
FONT_NAME    = "NotoSansKR"   # 우리가 정하는 등록용 이름

SEED = None  # 재현성 필요 시 42 같은 정수

# 텍스트 좌표(단위 pt). (0,0)=좌상단, y 증가=아래
FIELD_POS = {
    "name":       (140, 215),  # 환자명
    "dob":        (140, 245),  # 생년월일
    "date":       (450, 110),  # 발행일/처방일
    "medication": (60,  380),  # 약품명
    "dosage":     (420, 400),  # 복용법
    "code":       (82,  270),  # 질병코드(공백 삽입된 표기)
}

# -------- 유틸 --------
def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)

def sanitize_filename(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^\w\-\.가-힣]", "", s)
    return s[:60]

def load_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: 리스트가 비었거나 형식이 리스트가 아닙니다.")
    return data

def pick_one(lst):
    return random.choice(lst) if lst else ""

def parse_name_dob(s: str):
    """'이름, YYYY-MM-DD' 또는 주민번호 형식이 섞여 있어도 이름/생일 분리."""
    parts = [p.strip() for p in re.split(r"\s*,\s*", s, maxsplit=1)]
    # 주민번호(######-#######) 우선 탐지
    if len(parts) == 2 and re.fullmatch(r"\d{6}-\d{7}", parts[1]):
        return parts[0], parts[1]
    # YYYY-MM-DD 탐지
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        dob = m.group(1)
        name = s.replace(dob, "").replace(",", " ").strip()
        return name, dob
    return s.strip(), ""

def register_font(page) -> str:
    """
    PyMuPDF 1.26.x: insert_font(fontname=..., fontfile=...) 사용.
    실패 시 fontbuffer 방식으로 한 번 더 시도. 둘 다 실패하면 'helv' 반환.
    """
    try:
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"폰트 경로 없음: {FONT_PATH}")
        page.insert_font(fontname=FONT_NAME, fontfile=FONT_PATH)
        return FONT_NAME
    except Exception as e1:
        try:
            with open(FONT_PATH, "rb") as f:
                buf = f.read()
            page.insert_font(fontname=FONT_NAME, fontbuffer=buf)
            return FONT_NAME
        except Exception as e2:
            print(f"[WARN] 사용자 폰트 등록 실패: {repr(e1)} / {repr(e2)}")
            print("[INFO] 내장 폰트(helv)로 대체합니다.")
            return "helv"

# -------- 메인 --------
def main():
    if SEED is not None:
        random.seed(SEED)

    # JSON 경로
    meds_path     = os.path.join(FIELDS_DIR, "current_medication.json")
    people_path   = os.path.join(FIELDS_DIR, "patient_info.json")
    doses_path    = os.path.join(FIELDS_DIR, "dosage_info.json")
    codes_path    = os.path.join(FIELDS_DIR, "diagnosis_code.json")
    diseases_path = os.path.join(FIELDS_DIR, "disease_name.json")

    # 데이터 로드
    meds     = load_list(meds_path)
    people   = load_list(people_path)
    doses    = load_list(doses_path)
    codes    = load_list(codes_path)
    diseases = load_list(diseases_path)

    # 여러 건 생성 설정
    ensure_dir(OUT_DIR)
    batch_count = 3  # 원하는 개수
    today = datetime.now().strftime("%Y-%m-%d")
    today_compact = today.replace("-", "")

    # 코드-질병명 동기화: 같은 인덱스로 선택
    min_len = min(len(codes), len(diseases))
    if len(codes) != len(diseases):
        print(f"[WARN] 리스트 길이 불일치: codes={len(codes)}, diseases={len(diseases)} → min_len={min_len}")

    for i in range(1, batch_count + 1):
        # 무작위 선택
        idx = random.randrange(min_len)
        code_real       = codes[idx]          # 예: "J06"
        disease_matched = diseases[idx]       # 예: "급성 상기도 감염"

        med_selected    = pick_one(meds)
        person_selected = pick_one(people)
        dose_selected   = pick_one(doses)

        # 코드 PDF 표기용(글자 사이 공백)
        code_selected   = "   ".join(code_real)

        # 이름/생년월일
        name, dob = parse_name_dob(person_selected)

        # PDF에 찍을 값
        payload = {
            "name":       name,
            "dob":        dob,
            "date":       today,
            "medication": med_selected,
            "dosage":     dose_selected,
            "code":       code_selected,  # 공백 포함
        }

        # 로그/JSON 저장용(원본 코드/질병명)
        printload = {
            "name":         name,
            "dob":          dob,
            "date":         today,
            "medication":   med_selected,
            "dosage":       dose_selected,
            "code":         code_real,
            "disease_name": disease_matched,
        }

        # 파일명 스템(같은 이름으로 pdf/json 짝 저장)
        stem = f"Prescription_{today_compact}_{i:03d}_{sanitize_filename(name)}"
        out_pdf  = os.path.join(OUT_DIR, stem + ".pdf")
        out_json = os.path.join(OUT_DIR, stem + ".json")

        # PDF 생성
        doc = fitz.open(TEMPLATE_PDF)
        page = doc[0]
        fontname = register_font(page)

        for key, (x, y) in FIELD_POS.items():
            text = str(payload.get(key, "") or "")
            page.insert_text((x, y), text, fontname=fontname, fontsize=11)

        doc.save(out_pdf)
        doc.close()

        # JSON 생성(한 건당 1 파일)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(printload, f, ensure_ascii=False, indent=2)

        # 콘솔 로그
        # print(f"✅ Saved: {out_pdf}")
        # for k, v in printload.items():
        #     print(f"- {k}: {v}")
        # print("-" * 40)

if __name__ == "__main__":
    main()
