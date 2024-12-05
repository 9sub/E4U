# 환자 통증 점수 (1~10)

severity_weights = {
    'Calculus': 2,
    'Caries': 4,
    'CaS': 4,
    'CoS': 5,
    'Gingivitis': 6,
    'Gum': 9,
    'Hypodontia': 7,
    'MC': 10,  # 구강암
    'MouthUlcer': 2,
    'OLP': 5,
    'ToothDiscoloration': 3
}

# 점수 계산 함수
def calculate_danger_score(data, pain_score, bonus_per_symptom=5):
    total_score = 100
    total_symptoms = 0

    # 통증 점수 가중치 (1~10 정규화)
    pain_weight = pain_score / 10  # 0~1 범위
    print(data)
    # 치아 질환 점수 계산
    for tooth_num, diseases in data["tooth_diseases"].items():
        for disease in diseases:
            severity = severity_weights.get(disease["disease_name"], 1)
            confidence = disease.get("conf", 1)  # confidence가 없으면 기본값 1
            total_score -= severity * confidence *1.2
            total_symptoms += 1
    
    print(total_score, total_symptoms)

    # 잇몸 질환 점수 계산
    for region, diseases in data["gum_diseases"].items():
        for disease in diseases:
            severity = severity_weights.get(disease["disease_name"], 1)
            confidence = disease.get("conf", 1)  # confidence가 없으면 기본값 1
            total_score -= severity * confidence *1.2
            total_symptoms += 1

    print(total_score, total_symptoms)

    for region, diseases in data["etc_diseases"].items():
        for disease in diseases:
            severity = severity_weights.get(disease["disease_name"], 1)
            confidence = disease.get("confidence", 1)  # confidence가 없으면 기본값 1
            total_score -= severity * confidence * 0.9
            total_symptoms += 1

    print(total_score, total_symptoms)
    # 증상 개수에 따른 추가 보너스
    total_score -= total_symptoms * bonus_per_symptom

    # 통증 점수 반영
    total_score *= (1 - pain_weight)

    # 100점 만점으로 정규화
    normalized_score = max(total_score, 0)  # 최대 100점으로 제한
    print('total_score:', normalized_score)
    return round(normalized_score, 2)