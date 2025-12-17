import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm 

# ----------------------------
# 1. 경로 설정 및 파라미터
# ----------------------------
DATA_BASE_PATH = Path("/home/jwlee/volume/Qwen2-vl-finetune-wo/data/ko_zh_datasets_4")

# 원본 및 대상 파일/폴더 경로
TEST_JSON_PATH = DATA_BASE_PATH / "test_zh_ko.json"
TEST_IMG_DIR = DATA_BASE_PATH / "test"
VAL_IMG_DIR = DATA_BASE_PATH / "val"
VAL_JSON_PATH = DATA_BASE_PATH / "val_zh_ko.json"

VAL_RATIO_FROM_TEST = 0.5 # 50% 분할
random.seed(42) 

# --- 유틸리티 함수 (생략) ---
def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """JSON 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(file_path: Path, data: List[Dict[str, Any]]):
    """JSON 파일을 저장합니다."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
# ----------------------------

def split_test_to_val_and_test_and_update_paths():
    
    # 1. 이전 Val 폴더 제거 및 새로 생성 (재실행 시 안전 확보)
    if VAL_IMG_DIR.exists():
        shutil.rmtree(VAL_IMG_DIR)
    VAL_IMG_DIR.mkdir()

    print(f"--- 📂 JSON 파일 로드: {TEST_JSON_PATH.name} ---")
    
    try:
        test_data = load_json(TEST_JSON_PATH)
    except Exception as e:
        print(f"❌ JSON 파일 로드 중 오류 발생: {e}. 파일을 확인하세요.")
        return

    total_test_count = len(test_data)
    print(f"✅ 초기 test_zh_ko.json 항목 개수 (JSON 기준): {total_test_count:,}개")
    
    if total_test_count != 965:
        print(f"❗경고: JSON 항목 개수가 예상(965개)과 다릅니다. 현재 {total_test_count}개로 진행합니다.")

    # 2. 데이터 분할 (JSON 항목)
    random.shuffle(test_data)
    val_count = int(total_test_count * VAL_RATIO_FROM_TEST)
    
    val_set = test_data[:val_count]
    new_test_set = test_data[val_count:] 

    print(f"\n--- 📝 항목 분할 결과 (Validation/Test) ---")
    print(f"  > 분할된 Validation 셋 항목 개수: {len(val_set):,}개")
    print(f"  > 잔여 Test 셋 항목 개수: {len(new_test_set):,}개")
    
    # 3. 이미지 파일 이동 및 JSON 경로 업데이트
    print(f"\n--- 🏞️ 이미지 파일 이동/복사 및 JSON 경로 업데이트 ---")
    
    # 3.1. Validation 셋 처리: 이미지 이동 및 경로 업데이트
    moved_image_count = 0
    updated_val_set = []
    
    for item in tqdm(val_set, desc="Validation set processing"):
        # 이미지 파일 이름 추출 (예: '00427_zh.jpg')
        image_filename = Path(item["image"]).name 
        
        # 원본 경로는 현재 TEST_IMG_DIR 내에 있습니다.
        src_path = TEST_IMG_DIR / image_filename
        dst_path = VAL_IMG_DIR / image_filename
        
        # 파일 이동 (Test -> Val)
        if src_path.exists():
            shutil.move(src_path, dst_path)
            
            # JSON 항목의 'image' 경로를 새로운 'val' 경로로 업데이트
            # 경로 형식: 'data/ko_zh_datasets_3/val/파일이름.jpg' (최상위 폴더 기준)
            item["image"] = str(dst_path.relative_to(DATA_BASE_PATH.parent.parent)) 
            
            updated_val_set.append(item)
            moved_image_count += 1
        else:
            # 이전에 누락된 이미지가 없다고 했으므로, 이 경고는 이전 실행의 잔여 파일 문제일 수 있습니다.
            print(f"\n❌ 경고: 이미지 파일이 Test 폴더에 없습니다: {src_path}. 이 항목은 JSON에서도 제외됩니다.")

    # 3.2. New Test 셋 처리: JSON 경로만 업데이트
    updated_test_set = []
    
    for item in new_test_set:
        # 파일 이름 추출
        image_filename = Path(item["image"]).name
        src_path = TEST_IMG_DIR / image_filename
        
        # JSON 항목의 'image' 경로를 새로운 'test' 경로로 업데이트
        # 경로 형식: 'data/ko_zh_datasets_3/test/파일이름.jpg'
        item["image"] = str(src_path.relative_to(DATA_BASE_PATH.parent.parent))
        updated_test_set.append(item)

    # 4. 최종 JSON 파일 저장
    save_json(VAL_JSON_PATH, updated_val_set)
    save_json(TEST_JSON_PATH, updated_test_set)
    
    print(f"\n✅ val_zh_ko.json 저장 완료: ({len(updated_val_set):,}개 항목)")
    print(f"✅ test_zh_ko.json 업데이트 완료: ({len(updated_test_set):,}개 항목)")
    
    print("-" * 50)
    print(f"✅ 총 {moved_image_count:,}개 이미지 파일이 Val 폴더로 이동 완료.")
    print(f"✅ 최종 JSON 항목 기준 이미지 개수 (Val): {len(os.listdir(VAL_IMG_DIR)):,}개")
    print(f"✅ 최종 JSON 항목 기준 이미지 개수 (Test): {len(updated_test_set):,}개 (남아있는 총 파일 수와 다를 수 있음)")
    print("-" * 50)


if __name__ == "__main__":
    split_test_to_val_and_test_and_update_paths()