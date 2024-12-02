# 변환 함수
def return_json_format(data):
    result = {"tooth_diseases": {}, "gum_diseases": {}}
    if not data:
        return result

    # Tooth diseases 변환
    for tooth_num, diseases in data["tooth_diseases"].items():
        result["tooth_diseases"][tooth_num] = [
            {
                "disease_id": disease["disease_id"],
                "disease_name": disease["disease_name"],
                "conf": disease.get("confidence", 0)  # conf 값 포함
            }
            for disease in diseases
        ]

    # Gum diseases 변환
    for region, diseases in data["gum_diseases"].items():
        result["gum_diseases"][region] = [
            {
                "disease_id": disease["disease_id"],
                "disease_name": disease["disease_name"],
                "conf": disease.get("confidence", 0)  # conf 값 포함
            }
            for disease in diseases
        ]

    return result