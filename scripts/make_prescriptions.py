# make_prescriptions.py
import json, os, random, re
from datetime import datetime
import fitz  # PyMuPDF

# ========= 기본 설정 =========
BASE_DIR   = os.path.dirname(__file__)
FIELDS_DIR = os.path.join(BASE_DIR, "prescriptions/fields")
TEMPLATE_PDF = os.path.join(BASE_DIR, "Prescription_Template.pdf")

# 출력 폴더
OUT_DIR = os.path.join(BASE_DIR, "test-out-2")
PDF_DIR  = os.path.join(OUT_DIR, "pdf")
JSON_DIR = os.path.join(OUT_DIR, "annotation_json")
IMG_DIR  = os.path.join(OUT_DIR, "images")

# 폰트
FONT_PATH    = os.path.abspath(os.path.join(BASE_DIR, "NotoSansKR-Regular.ttf"))
FONT_NAME    = "NotoSansKR"

SEED = None  # 재현성 필요 시 42 같은 정수

# 텍스트 좌표
FIELD_POS = {
    "name":       (140, 215),
    "dob":        (140, 245),
    "date":       (450, 110),
    "medication_A": (60,  380),
    "medication_B": (60,  400),
    "medication_C": (60,  420),
    "medication_D": (60,  440),
    "dosage":     (420, 400),
    "code_A":     (82,  270),
    "code_B":     (82,  300),
    "hospital":   (400, 180),
    "period_A_1":   (300, 380),
    "period_A_2":   (345, 380),
    "period_A_3":   (390, 380),
    "period_B_1":   (300, 400),
    "period_B_2":   (345, 400),
    "period_B_3":   (390, 400),
    # "period_C_1":   (300, 420),
    # "period_C_2":   (345, 420),
    # "period_C_3":   (390, 420),
    # "period_D_1":   (300, 440),
    # "period_D_2":   (345, 440),
    # "period_D_3":   (390, 440)
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
    meds_A_path     = os.path.join(FIELDS_DIR, "current_medication.json")
    meds_B_path     = os.path.join(FIELDS_DIR, "current_medication.json")
    # meds_C_path     = os.path.join(FIELDS_DIR, "current_medication.json")
    # meds_D_path     = os.path.join(FIELDS_DIR, "current_medication.json")
    
    people_path   = os.path.join(FIELDS_DIR, "patient_info.json")
    doses_path    = os.path.join(FIELDS_DIR, "dosage_info.json")
    codes_A_path    = os.path.join(FIELDS_DIR, "diagnosis_code.json")
    codes_B_path    = os.path.join(FIELDS_DIR, "diagnosis_code.json")
    # codes_C_path    = os.path.join(FIELDS_DIR, "diagnosis_code.json")
    # codes_D_path    = os.path.join(FIELDS_DIR, "diagnosis_code.json")
    
    diseases_path = os.path.join(FIELDS_DIR, "disease_name.json")
    hospital_path = diseases_path = os.path.join(FIELDS_DIR, "hospital_info.json")
    
    us_period_A_1_path = os.path.join(FIELDS_DIR, "usage_period_1.json")
    us_period_A_2_path = os.path.join(FIELDS_DIR, "usage_period_2.json")
    us_period_A_3_path = os.path.join(FIELDS_DIR, "usage_period_3.json")
    us_period_B_1_path = os.path.join(FIELDS_DIR, "usage_period_1.json")
    us_period_B_2_path = os.path.join(FIELDS_DIR, "usage_period_2.json")
    us_period_B_3_path = os.path.join(FIELDS_DIR, "usage_period_3.json")
    # us_period_C_1_path = os.path.join(FIELDS_DIR, "usage_period_1.json")
    # us_period_C_2_path = os.path.join(FIELDS_DIR, "usage_period_2.json")
    # us_period_C_3_path = os.path.join(FIELDS_DIR, "usage_period_3.json")
    # us_period_D_1_path = os.path.join(FIELDS_DIR, "usage_period_1.json")
    # us_period_D_2_path = os.path.join(FIELDS_DIR, "usage_period_2.json")
    # us_period_D_3_path = os.path.join(FIELDS_DIR, "usage_period_3.json")
    
    
    # 데이터 로드
    meds_A     = load_list(meds_A_path)
    meds_B     = load_list(meds_B_path)
    # meds_C     = load_list(meds_C_path)
    # meds_D     = load_list(meds_D_path)
    
    people   = load_list(people_path)
    doses    = load_list(doses_path)
    codes_A    = load_list(codes_A_path)
    codes_B    = load_list(codes_B_path)
    # codes_C    = load_list(codes_C_path)
    # codes_D    = load_list(codes_D_path)
    
    diseases = load_list(diseases_path)
    hospital = load_list(hospital_path)
    period_A_1 = load_list(us_period_A_1_path)
    period_A_2 = load_list(us_period_A_2_path)
    period_A_3 = load_list(us_period_A_3_path)
    period_B_1 = load_list(us_period_B_1_path)
    period_B_2 = load_list(us_period_B_2_path)
    period_B_3 = load_list(us_period_B_3_path)
    # period_C_1 = load_list(us_period_C_1_path)
    # period_C_2 = load_list(us_period_C_2_path)
    # period_C_3 = load_list(us_period_C_3_path)
    # period_D_1 = load_list(us_period_D_1_path)
    # period_D_2 = load_list(us_period_D_2_path)
    # period_D_3 = load_list(us_period_D_3_path)
    

    # 폴더 생성
    for d in [PDF_DIR, JSON_DIR, IMG_DIR]:
        ensure_dir(d)

    batch_count = 3 # 원하는 개수
    today = datetime.now().strftime("%Y-%m-%d")
    today_compact = today.replace("-", "")

    min_len_A = min(len(codes_A), len(diseases))
    min_len_B = min(len(codes_B), len(diseases))
    # min_len_C = min(len(codes_C), len(diseases))
    # min_len_D = min(len(codes_D), len(diseases))
    
    
    if len(codes_A) != len(diseases):
        print(f"[WARN] 리스트 길이 불일치: codes={len(codes_A)}, diseases={len(diseases)} → min_len={min_len_A}")

    if len(codes_B) != len(diseases):
        print(f"[WARN] 리스트 길이 불일치: codes={len(codes_B)}, diseases={len(diseases)} → min_len={min_len_B}")


    for i in range(1, batch_count + 1):
        idx_A = random.randrange(min_len_A)
        idx_B = random.randrange(min_len_B)
        # idx_C = random.randrange(min_len_C)
        # idx_D = random.randrange(min_len_D)
        
        
        code_A_real       = codes_A[idx_A]
        code_B_real       = codes_B[idx_B]
        # code_C_real       = codes_C[idx_C]
        # code_D_real       = codes_C[idx_D]
        
        disease_matched_A = diseases[idx_A]
        disease_matched_B = diseases[idx_B]
        # disease_matched_C = diseases[idx_C]
        # disease_matched_D = diseases[idx_D]

        med_A_selected    = pick_one(meds_A)
        med_B_selected    = pick_one(meds_B)
        # med_C_selected    = pick_one(meds_C)
        # med_D_selected    = pick_one(meds_D)
        
        
        person_selected = pick_one(people)
        dose_selected   = pick_one(doses)
        hospital_selected = pick_one(hospital)
        
        period_A_1_selected = pick_one(period_A_1)
        period_A_2_selected = pick_one(period_A_2)
        period_A_3_selected = pick_one(period_A_3)
        period_B_1_selected = pick_one(period_B_1)
        period_B_2_selected = pick_one(period_B_2)
        period_B_3_selected = pick_one(period_B_3)
        # period_C_1_selected = pick_one(period_C_1)
        # period_C_2_selected = pick_one(period_C_2)
        # period_C_3_selected = pick_one(period_C_3)
        # period_D_1_selected = pick_one(period_D_1)
        # period_D_2_selected = pick_one(period_D_2)
        # period_D_3_selected = pick_one(period_D_3)

        code_A_selected   = "   ".join(code_A_real)
        code_B_selected   = "   ".join(code_B_real)
        # code_C_selected   = "   ".join(code_C_real)
        # code_D_selected   = "   ".join(code_D_real)
        
        
        name, dob = parse_name_dob(person_selected)

        payload = {
            "name":       name,
            "dob":        dob,
            "date":       today,
            "medication_A": med_A_selected,
            "medication_B": med_B_selected,
            # "medication_C": med_C_selected,
            # "medication_D": med_D_selected,
            "dosage":     dose_selected,
            "code_A":       code_A_selected,
            "code_B":       code_B_selected,
            # "code_C":       code_C_selected,
            # "code_D":       code_D_selected,
            "hospital":   hospital_selected,
            "period_A_1":   period_A_1_selected,
            "period_A_2":   period_A_2_selected,
            "period_A_3":   period_A_3_selected,
            "period_B_1":   period_B_1_selected,
            "period_B_2":   period_B_2_selected,
            "period_B_3":   period_B_3_selected,
            # "period_C_1":   period_C_1_selected,
            # "period_C_2":   period_C_2_selected,
            # "period_C_3":   period_C_3_selected,
            # "period_D_1":   period_D_1_selected,
            # "period_D_2":   period_D_2_selected,
            # "period_D_3":   period_D_3_selected
        }

        printload = {
            "name":         name,
            "dob":          dob,
            "date":         today,
            "medication_A":   med_A_selected,
            "medication_B":   med_B_selected,
            # "medication_C":   med_C_selected,
            # "medication_D": med_D_selected,
            "dosage":       dose_selected,
            "code_A":         code_A_real,
            "code_B":         code_B_real,
            # "code_C":         code_C_real,
            # "code_D":         code_D_real,
            "disease_name_A": disease_matched_A,
            "disease_name_B": disease_matched_B,
            "hospital":   hospital_selected,
            "period_A_1":   period_A_1_selected,
            "period_A_2":   period_A_2_selected,
            "period_A_3":   period_A_3_selected,
            "period_B_1":   period_B_1_selected,
            "period_B_2":   period_B_2_selected,
            "period_B_3":   period_B_3_selected,
            # "period_C_1":   period_C_1_selected,
            # "period_C_2":   period_C_2_selected,
            # "period_C_3":   period_C_3_selected,
            # "period_D_1":   period_D_1_selected,
            # "period_D_2":   period_D_2_selected,
            # "period_D_3":   period_D_3_selected
            
        }

        # 파일명 5자리 숫자 포맷
        stem = f"{i:05d}"
        out_pdf  = os.path.join(PDF_DIR, stem + ".pdf")
        out_json = os.path.join(JSON_DIR, stem + ".json")
        out_jpg  = os.path.join(IMG_DIR, stem + ".jpg")

        # PDF 생성
        doc = fitz.open(TEMPLATE_PDF)
        page = doc[0]
        fontname = register_font(page)

        for key, (x, y) in FIELD_POS.items():
            text = str(payload.get(key, "") or "")
            page.insert_text((x, y), text, fontname=fontname, fontsize=11)

        doc.save(out_pdf)
        doc.close()

        # PDF → JPG 변환
        doc = fitz.open(out_pdf)
        page = doc[0]
        pix = page.get_pixmap()
        pix.save(out_jpg)
        doc.close()

        # JSON 저장
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(printload, f, ensure_ascii=False, indent=2)

        # 로그
        print(f"✅ Saved: {out_pdf}, {out_jpg}, {out_json}")
        for k, v in printload.items():
            print(f"- {k}: {v}")
        print("-" * 40)

if __name__ == "__main__":
    main()
