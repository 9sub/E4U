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

    tooth_disease_names = [
        translate_disease(d.disease_name)
        for diseases in data.tooth_diseases.values()
        for d in diseases
    ]

    tooth_disease_names_str = ", ".join(sorted(set(tooth_disease_names))) if tooth_disease_names else "질환없음"

    # 잇몸 질병 문자열 생성
    gum_diseases_str = ", ".join(
        [f"{region} : {', '.join(translate_disease(d.disease_name) for d in diseases)}"
         for region, diseases in data.gum_diseases.items() if diseases]
    )

    gum_diseases_names = [
        translate_disease(d.disease_name)
        for diseases in data.gum_diseases.values()
        for d in diseases
    ]

    gum_diseases_names_str = ", ".join(sorted(set(gum_diseases_names))) if gum_diseases_names else "질환없음"
    

    # 기타 질병 문자열 생성
    etc_diseases_str = ", ".join(
        [f"{region} : {', '.join(translate_disease(d.disease_name) for d in diseases)}"
         for region, diseases in data.etc_diseases.items() if diseases]
    )

    etc_diseases_names = [
        translate_disease(d.disease_name)
        for diseases in data.etc_diseases.values()
        for d in diseases
    ]

    etc_diseases_names_str = ", ".join(sorted(set(etc_diseases_names))) if etc_diseases_names else "질환없음"



    # 환자 통증 위치
    symptom_area_str = ", ".join(data.symptomArea)

    if not symptom_area_str.strip():
        symptom_area_str = "통증위치없음"

    # 환자의 증상
    symptom_text_str = ", ".join(data.symptomText)

    if not symptom_text_str.strip():
        symptom_text_str = "증상없음"

    # 환자 통증 정도
    pain_level_str = str(data.painLevel)

    # 최종 변환 결과
    # result = f"""
    # 예측된 구강질환: < {tooth_diseases_str} , {gum_diseases_str} >
    # """
    
    # detailed_result = f"""
    # 예측된 구강질환: < {tooth_diseases_str} , {gum_diseases_str} >
    # 환자의 증상: < {symptom_text_str} >
    # 환자 통증 위치: < {symptom_area_str} >
    # """

    tooth_disease = f"""
    < {tooth_diseases_str}, >
    """

    gum_disease = f"""
    < {gum_diseases_str} >
    """

    etc_disease = f"""
    < {etc_diseases_str} >
    """

    return tooth_disease.strip(), gum_disease.strip(),etc_disease.strip(), symptom_area_str, tooth_disease_names_str , gum_diseases_names_str, etc_diseases_names_str

'''
    잇몸질병: < {gum_diseases_str} >
    환자 통증 위치: < {symptom_area_str} >
    환자 통증 정도: < {pain_level_str} >
    환자의 증상: < {symptom_text_str} >
'''