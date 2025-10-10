# make_prescriptions.py
import json, os, random, re
from datetime import datetime
import fitz  # PyMuPDF

# ========= 기본 설정 =========
BASE_DIR   = os.path.dirname(__file__)
FIELDS_DIR = os.path.join(BASE_DIR, "fields")

TEMPLATE_PDF = os.path.join(BASE_DIR, "Prescription_Template.pdf")
OUTPUT_PDF   = os.path.join(BASE_DIR, "Prescription_out.pdf")

# 폰트: 동일 폴더에 둔 NotoSansKR-Regular.ttf 사용 (한글/영문 OK)
FONT_PATH    = os.path.abspath(os.path.join(BASE_DIR, "NotoSansKR-Regular.ttf"))
FONT_NAME    = "NotoSansKR"   # 우리가 정하는 등록용 이름

SEED = None  # 재현성 필요하면 42 같은 정수

# 텍스트 좌표(단위 pt). (0,0)=좌상단, y 증가=아래
FIELD_POS = {
    "name":       (140, 215),  # 환자명
    "dob":        (140, 245),  # 생년월일
    "date":       (450, 110),  # 발행일/처방일
    "medication": (60, 380),  # 약품명
    "dosage":     (420, 400),  # 복용법
}

# -------- 유틸 --------
def load_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: 리스트가 비었거나 형식이 리스트가 아닙니다.")
    return data

def pick_one(lst):
    return random.choice(lst) if lst else ""

def parse_name_dob(s: str):
    """'이름, YYYY-MM-DD' -> (이름, 생년월일)"""
    parts = [p.strip() for p in re.split(r"\s*,\s*", s, maxsplit=1)]
    if len(parts) == 2 and re.fullmatch(r"\d{6}-\d{7}", parts[1]):
        return parts[0], parts[1]
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
        # 경로 체크
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"폰트 경로 없음: {FONT_PATH}")
        # 파일 경로로 등록
        page.insert_font(fontname=FONT_NAME, fontfile=FONT_PATH)
        return FONT_NAME
    except Exception as e1:
        try:
            # 버퍼로 재시도
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
    meds_path   = os.path.join(FIELDS_DIR, "current_medication.json")
    people_path = os.path.join(FIELDS_DIR, "patient_info.json")
    doses_path  = os.path.join(FIELDS_DIR, "dosage_info.json")

    # 데이터 로드
    meds   = load_list(meds_path)
    people = load_list(people_path)
    doses  = load_list(doses_path)

    # 무작위 선택
    med_selected    = pick_one(meds)
    person_selected = pick_one(people)
    dose_selected   = pick_one(doses)

    name, dob = parse_name_dob(person_selected)
    today = datetime.now().strftime("%Y-%m-%d")

    payload = {
        "name":       name,
        "dob":        dob,
        "date":       today,
        "medication": med_selected,
        "dosage":     dose_selected,
    }

    # PDF 열기
    doc = fitz.open(TEMPLATE_PDF)
    page = doc[0]

    # ✅ 폰트 등록 (이름을 지정해서 등록하고 그대로 사용)
    fontname = register_font(page)

    # 텍스트 삽입
    for key, (x, y) in FIELD_POS.items():
        text = str(payload.get(key, "") or "")
        page.insert_text((x, y), text, fontname=fontname, fontsize=11)

    doc.save(OUTPUT_PDF)
    doc.close()

    print(f"✅ Saved: {OUTPUT_PDF}")
    print("== 사용된 데이터 ==")
    for k, v in payload.items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    main()
