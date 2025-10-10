# scripts/render_prescription.py
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
import json
import os

TEMPLATE_PATH = "data/synth_rx/prescriptions.jsonl"
OUTPUT_DIR = "data/synth_rx/pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_prescription(c, data):
    """하나의 처방전 데이터를 PDF 위에 출력"""
    c.setFont("Helvetica", 12)
    x, y = 20 * mm, 270 * mm

    c.drawString(x, y, f"처방전 유형: {data['prescription_type']}")
    c.drawString(x, y - 10, f"환자 정보: {data['patient_info']}")
    c.drawString(x, y - 20, f"진단 코드: {data['diagnosis_code']}")
    c.drawString(x, y - 30, f"질병명: {data['disease_name']}")
    c.drawString(x, y - 40, f"현재 복용 약물: {data['current_medication']}")
    c.drawString(x, y - 50, f"용량 정보: {data['dosage_info']}")
    c.drawString(x, y - 60, f"복용 기간: {data['usage_period']}")
    c.drawString(x, y - 70, f"보험: {data['insurance']}")
    c.drawString(x, y - 80, f"병원 정보: {data['hospital_info']}")

    c.showPage()

def main():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            save_path = os.path.join(OUTPUT_DIR, f"prescription_{i+1}.pdf")

            c = canvas.Canvas(save_path, pagesize=A4)
            draw_prescription(c, data)
            c.save()
            print(f"✅ {save_path} 저장 완료")

if __name__ == "__main__":
    main()
