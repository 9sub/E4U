# 환자 통증 점수 (1~10)
patient_pain_score = 7  # 환자가 제공한 통증 점수 (1~10)


severity_weights = {
    'Calculus': 2,
    'Caries': 2,
    'CaS': 3,
    'CoS': 4,
    'Gingivitis': 5,
    'Gum': 8,
    'Hypodontia': 6,
    'MC': 10,  # 구강암
    'MouthUlcer': 1,
    'OLP': 4,
    'ToothDiscoloration': 2
}

# 점수 계산 함수
def calculate_danger_score(data, pain_score, bonus_per_symptom=5):
    total_score = 0
    total_symptoms = 0

    # 통증 점수 가중치 (1~10 정규화)
    pain_weight = pain_score / 10  # 0~1 범위

    # 치아 질환 점수 계산
    for tooth_num, diseases in data["tooth_diseases"].items():
        for disease in diseases:
            severity = severity_weights.get(disease["disease_name"], 1)
            confidence = disease["confidence"]
            total_score += severity * confidence
            total_symptoms += 1

    # 잇몸 질환 점수 계산
    for region, diseases in data["gum_diseases"].items():
        for disease in diseases:
            severity = severity_weights.get(disease["disease_name"], 1)
            confidence = disease["confidence"]
            total_score += severity * confidence
            total_symptoms += 1

    # 증상 개수에 따른 추가 보너스
    total_score += total_symptoms * bonus_per_symptom

    # 통증 점수 반영
    total_score *= (1 + pain_weight)

    # 100점 만점으로 정규화
    normalized_score = 100-min(total_score, 100)  # 최대 100점으로 제한
    return round(normalized_score, 2)