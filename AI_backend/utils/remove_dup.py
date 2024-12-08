def remove_dup(results):
    """
    치아와 잇몸 질환에서 같은 종류의 질환이 있을 경우 신뢰도가 가장 높은 것만 남기는 함수
    
    Args:
        results (dict): {
            'tooth_diseases': {tooth_number: [disease_info, ...]}, 
            'gum_diseases': {region: [disease_info, ...]}
        }
    
    Returns:
        dict: 중복이 제거된 결과
    """
    def filter_diseases(diseases):
        # disease_id 기준으로 가장 confidence 높은 것만 남김
        max_confidences = {}
        for disease in diseases:
            disease_id = disease['disease_id']
            if disease_id not in max_confidences or disease['confidence'] > max_confidences[disease_id]['confidence']:
                max_confidences[disease_id] = disease
        return list(max_confidences.values())

    # 치아 질환 중복 제거
    filtered_tooth_diseases = {}
    for tooth_number, diseases in results['tooth_diseases'].items():
        filtered_tooth_diseases[tooth_number] = filter_diseases(diseases)

    # 잇몸 질환 중복 제거
    filtered_gum_diseases = {}
    for region, diseases in results['gum_diseases'].items():
        filtered_gum_diseases[region] = filter_diseases(diseases)

    return {
        'tooth_diseases': filtered_tooth_diseases,
        'gum_diseases': filtered_gum_diseases
    }