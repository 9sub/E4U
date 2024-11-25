def result_report_form(data):
    # 질병 이름 영한 변환 사전
    disease_translation = {
        "Calculus": "치석",
        "Caries": "충치",
        "CaS": "칸디다증",
        "CoS": "구순포진",
        "Gingivitis": "치은염",
        "GUM": "치주염",
        "Hypodontia": "치아 결손",
        "MC": "구강암",
        "MouthUlcer": "구내염",
        "OLP": "구강 편평태선",
        "ToothDiscoloration": "치아 변색"
    }

    # 영어 질병명을 한글로 변환하는 함수
    def translate_disease(disease_name):
        return disease_translation.get(disease_name, disease_name)

    # 치아 질병 문자열 생성
    tooth_diseases_str = ", ".join(
        [f"치아번호 {tooth_num} : {', '.join(translate_disease(d.disease_name) for d in diseases)}"
         for tooth_num, diseases in data.tooth_diseases.items()]
    )

    # 잇몸 질병 문자열 생성
    gum_diseases_str = ", ".join(
        [f"{region} : {', '.join(translate_disease(d.disease_name) for d in diseases)}"
         for region, diseases in data.gum_diseases.items() if diseases]
    )

    # 환자 통증 위치
    symptom_area_str = ", ".join(data.symptomArea)

    # 환자의 증상
    symptom_text_str = ", ".join(data.symptomText)

    # 환자 통증 정도
    pain_level_str = str(data.painLevel)

    # 최종 변환 결과
    result = f"""
    예측된 구강질환: < {tooth_diseases_str} , {gum_diseases_str} >
    """
    
    detailed_result = f"""
    예측된 구강질환: < {tooth_diseases_str} , {gum_diseases_str} >
    환자의 증상: < {symptom_text_str} >
    """
    return result.strip(), detailed_result.strip()

'''
    잇몸질병: < {gum_diseases_str} >
    환자 통증 위치: < {symptom_area_str} >
    환자 통증 정도: < {pain_level_str} >
    환자의 증상: < {symptom_text_str} >
'''