# 질병 이름의 다양한 표현을 매핑
disease_name_mapping = {
    "Calculus": ["치석"],
    "Caries": ["충치", "치아우식증","치아 우식증"],
    "CaS": ["칸디다증","칸디다성","칸디다성 구내염"],
    "CoS": ["구순포진"],
    "Gingivitis": ["치은염"],
    "GUM": ["치주염"],
    "Hypodontia": ["치아 결손"],
    "MC": ["구강암","구순암"],
    "MouthUlcer": ["구내염","구순염","구순포진","헤르페스"],
    "OLP": ["구강 편평태선"],
    "ToothDiscoloration": ["치아 변색"]
}

# 문자열에서 질병 이름 추출
def extract_disease_names(output):
    found_diseases = []
    for disease, synonyms in disease_name_mapping.items():
        for synonym in synonyms:
            if synonym in output:
                found_diseases.append(disease)
                break  # 하나의 질병명만 추가하고 다음 질병으로 넘어감
    return found_diseases


class_map = {
    0: 'Calculus', 
    1: 'Caries', 
    2: 'CaS', 
    3: 'CoS', 
    4: 'Gingivitis', 
    5: 'Gum', 
    6: 'Hypodontia', 
    7: 'MC', 
    8: 'MouthUlcer', 
    9: 'OLP', 
    10: 'ToothDiscoloration'
}

def adjust_and_weight_conf(data, target_locations, disease_names):
    # Tooth diseases 가중치 증가
    for tooth_id in target_locations['tooth_disease']:
        if tooth_id in data['tooth_diseases']:
            for entry in data['tooth_diseases'][tooth_id]:
                if entry['disease_name'] in disease_names:
                    #print(entry['disease_name'], entry['confidence'])
                    entry['confidence'] *= 1.2  # conf 값 1.2배 증가
                    #print(entry['disease_name'], entry['confidence'])

    # Gum diseases 가중치 증가
    for gum_location in target_locations['gum_diseases']:
        if gum_location in data['gum_diseases']:
            for entry in data['gum_diseases'][gum_location]:
                if entry['disease_name'] in disease_names:
                    entry['confidence'] *= 1.2  # conf 값 1.2배 증가

    #conf < 0.35 필터링
    data['tooth_diseases'] = {
        tooth_id: [
            entry for entry in entries if entry['confidence'] >= 0.35
        ]
        for tooth_id, entries in data['tooth_diseases'].items()
    }
    data['gum_diseases'] = {
        gum_location: [
            entry for entry in entries if entry['confidence'] >= 0.35
        ]
        for gum_location, entries in data['gum_diseases'].items()
    }

    return data