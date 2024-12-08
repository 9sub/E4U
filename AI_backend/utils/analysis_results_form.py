
def analysis_results_form(results):
    disease_name_mapping = {
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

    result = ""
    if(results == {}):
        return
    if(results['tooth_diseases'] == {} and results['gum_diseases'] == {}):
        return "<질환 없음>"
    if(results['tooth_diseases'] != {}):
        for tooth_num, diseases in results['tooth_diseases'].items():
            for disease in diseases:
                korean_name = disease_name_mapping.get(disease['disease_name'], disease['disease_name'])
                result += f"tooth_num {tooth_num}: disease_name: {korean_name}, "

    if(results['gum_diseases'] != {}):
        for region, diseases in results['gum_diseases'].items():
            for disease in diseases:
                korean_name = disease_name_mapping.get(disease['disease_name'], disease['disease_name'])
                result += f"region {region}: disease_name: {korean_name}, "

    result = "<"+ result + ">"
    print(result)
    return result
