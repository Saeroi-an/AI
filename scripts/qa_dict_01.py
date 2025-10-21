import codecs
import json
import os
from qa_datasets_LLaVA_en import generate_conversation, add_image_tag_to_conversation

if __name__ == "__main__":
    
    skipped_files_not_dict = [
        '00715.json', '00648.json', '00403.json', '00789.json', '00145.json', '00097.json', '00032.json', 
        '00728.json', '00010.json', '00724.json', '00310.json', '00673.json', '00637.json', '00523.json', 
        '00535.json', '00618.json', '00315.json', '00675.json', '00071.json', '00695.json', '00055.json',
        '00530.json', '00771.json', '00007.json', '00225.json', '00692.json', '00662.json', '00405.json',
        '00654.json', '00537.json', '00290.json', '00656.json', '00509.json', '00136.json', '00204.json', 
        '00314.json', '00698.json', '00227.json', '00168.json', '00380.json', '00363.json', '00276.json',
        '00734.json', '00507.json', '00506.json', '00353.json', 
        '00463.json', '00682.json', '00444.json', '00489.json', '00307.json', '00658.json', '00792.json',
        '00244.json', '00531.json', '00348.json', '00572.json', '00670.json', '00396.json', '00636.json',
        '00790.json', '00638.json', '00447.json', '00073.json', '00139.json', '00486.json', '00446.json', 
        '00494.json', '00412.json', '00627.json', '00719.json', '00339.json', '00248.json', '00758.json',
        '00078.json', '00793.json', '00676.json', '00038.json', '00382.json', '00519.json', '00155.json', 
        '00615.json', '00170.json', '00746.json', '00327.json', '00651.json', '00222.json', '00591.json', 
        '00607.json', '00224.json', '00644.json', '00414.json', '00229.json', '00286.json', '00378.json',
        '00390.json', '00729.json', '00046.json', '00679.json', '00102.json', '00745.json', '00114.json',
        '00067.json', '00250.json', '00629.json', '00628.json', '00330.json', '00216.json', '00721.json', 
        '00338.json', '00713.json', '00096.json', '00541.json', '00333.json', '00434.json', '00795.json',
        '00620.json', '00372.json', '00291.json', '00042.json', '00784.json', '00579.json', 
        '00198.json', '00037.json', '00770.json', '00141.json', '00580.json', '00705.json', '00148.json',
        '00669.json', '00249.json', '00282.json', '00023.json', '00306.json', '00526.json', '00123.json', 
        '00377.json', '00471.json', '00625.json', '00074.json', '00164.json', '00326.json', '00113.json',
        '00223.json', '00350.json', '00665.json', '00694.json', '00273.json', '00485.json', '00039.json', 
        '00424.json', '00736.json', '00166.json', '00150.json', '00553.json', '00358.json', '00151.json',
        '00672.json', '00242.json', '00431.json', '00524.json', '00087.json', '00384.json', '00174.json',
        '00173.json', '00730.json', '00495.json', '00025.json', '00202.json', '00086.json', '00100.json',
        '00518.json', '00129.json', '00169.json', '00165.json', '00147.json', '00288.json', '00026.json',
        '00566.json', '00081.json', '00035.json', '00451.json', '00488.json', '00345.json', '00543.json', 
        '00387.json', '00504.json', '00415.json', '00441.json', '00634.json', '00723.json', '00034.json', '00082.json', '00186.json', '00473.json', '00798.json', '00127.json', '00130.json', '00240.json', '00762.json', '00236.json', '00462.json', '00763.json', '00062.json', '00632.json', '00090.json', '00185.json', '00554.json', '00183.json', '00460.json', '00022.json', '00101.json', '00474.json', '00663.json', '00453.json', '00457.json', '00536.json', '00775.json', '00089.json', '00769.json', '00189.json', '00253.json', '00420.json', '00004.json', 
        '00532.json', '00467.json', '00153.json', '00464.json', '00458.json', '00296.json', '00259.json',
        '00490.json', '00608.json', '00320.json', '00258.json', '00268.json', '00714.json', '00237.json', 
        '00749.json', '00116.json', '00228.json', '00484.json', '00799.json', '00765.json', '00318.json', '00410.json', '00639.json', '00557.json', '00483.json', '00323.json', '00443.json', '00559.json', '00402.json', '00712.json', '00710.json', '00076.json', '00399.json', '00525.json', '00428.json', '00529.json', '00027.json', '00786.json', '00267.json', '00753.json', '00311.json', '00154.json', '00041.json', '00312.json', '00187.json', '00152.json', '00640.json', '00047.json', '00018.json', '00028.json', '00201.json', '00172.json', '00780.json', '00657.json', '00108.json', '00287.json', '00251.json', '00126.json', '00737.json', '00271.json', '00602.json', '00576.json', '00703.json', '00182.json', '00550.json', '00470.json', '00452.json', '00755.json', '00247.json', '00791.json', '00188.json', '00409.json', '00513.json', '00561.json', '00573.json', '00383.json', '00356.json', '00411.json', '00163.json', '00781.json', '00391.json', '00270.json', '00298.json', '00497.json', '00542.json', '00257.json', '00212.json', '00119.json', '00708.json', '00671.json', '00125.json', '00389.json', '00317.json', '00009.json', '00281.json', '00080.json', '00678.json', '00577.json', '00594.json', '00697.json', '00747.json', '00309.json', '00562.json', '00110.json', '00368.json', '00546.json', '00016.json', '00105.json', '00596.json', '00480.json', '00238.json', '00095.json', '00324.json', '00406.json', '00143.json', '00061.json', '00049.json', '00732.json', '00128.json', '00600.json', '00642.json']


    json_folder = "data/cord_sample/annotations_json"
    output_file = "synth_rx/llava_receipt_dataset_dict.json"

    llava_dataset = []
    skipped_files_included_sub = []
    skipped_1 = []

    for file_name in skipped_files_not_dict:
        json_path = os.path.join(json_folder, file_name)
        
        with codecs.open(json_path, "r", encoding="utf-8-sig") as f:
            receipt_json = json.load(f)
        
        image_filename = receipt_json.get("image", file_name.replace(".json", ".jpg"))
        menu_data = receipt_json["gt_parse"].get("menu", {})
        
        print(f"현재 파일명: {file_name}")

        # ✅ menu가 dict 한 개만 존재하면 리스트로 변환
        if isinstance(menu_data, dict):
            menu_items = [menu_data]
        elif isinstance(menu_data, list):
            menu_items = menu_data
        else:
            print(f"⚠️ menu가 dict/list 아님 → skip: {file_name}")
            continue

        valid_items = []
        for idx, item in enumerate(menu_items):
            if not isinstance(item, dict):
                print(f"⚠️ menu 내부 항목이 dict 아님 → skip: {file_name} index={idx}")
                skipped_1.append(file_name)
                continue

            # # ✅ price / unitprice 처리
            # if "unitprice" in item:
            #     cnt = int(item.get("cnt", "1"))  # cnt가 없으면 1로 처리
            #     unitprice = float(item["unitprice"].replace(".", ""))  # 가격 형식 "50.000" → 50000
            #     if "price" not in item:
            #         item["price"] = str(unitprice * cnt)  # cnt 곱해서 price 설정
            #     else:
            #         price = float(item["price"].replace(".", ""))
            #         if price != unitprice * cnt:
            #             print(f"⚠️ price/unitprice 불일치 ({file_name} index={idx}) → cnt 곱한 price로 교체")
            #             item["price"] = str(unitprice * cnt)
            #     item.pop("unitprice", None)
           
            # 2. price / unitprice 처리
            if "unitprice" in item:
                # price가 없으면 unitprice로 대체
                if "price" not in item:
                    item["price"] = item["unitprice"]
                else:
                #     # price가 있어도 unitprice와 다르면 그냥 unitprice로 덮어쓰기
                     if item["price"] != item["unitprice"]:
                         print(f"⚠️ price와 unitprice 값 불일치: {file_name} index {idx} -> price={item['price']}, unitprice={item['unitprice']}.")
                #         item["price"] = item["unitprice"]

                # unitprice 제거
                item.pop("unitprice", None)


            
            # 3. cnt 기본값 설정
            if "cnt" not in item:
                item["cnt"] = "1"

            # ✅ subkey 검사
            sub_keys = [k for k in item.keys() if k not in ["nm", "cnt", "price"]]
            if sub_keys:
                print(f"❤️ sub_key 존재 → skip: {file_name}, index={idx}, sub_keys={sub_keys}")
                skipped_files_included_sub.append(file_name)
                continue

            valid_items.append(item)

        if not valid_items:
            print(f"⚠️ 유효한 메뉴 없음 → skip: {file_name}")
            continue

        # conversation 생성
        used_items_global = set()
        num_convos = 1  # 필요 시 min/max 랜덤 적용 가능
        for aug_index in range(num_convos):
            conversation, used_items_global = generate_conversation(
                {"gt_parse": {"menu": valid_items}}, used_items_global
            )
            if conversation is None:
                continue

            conversation = add_image_tag_to_conversation(conversation)

            entry = {
                "id": f"{receipt_json.get('id', file_name.replace('.json',''))}_aug{aug_index}",
                "image": image_filename,
                "conversations": conversation
            }
            llava_dataset.append(entry)

    # 결과 저장
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(llava_dataset, f_out, ensure_ascii=False, indent=2)

    if skipped_files_included_sub:
        skipped_files_included_sub = list(set(skipped_files_included_sub))
        print("Skipped files due to sub keys:", skipped_files_included_sub)

    print(f"LLaVA dataset generated! 총 entries: {len(llava_dataset)}")