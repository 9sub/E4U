# FDI 번호와 클래스 인덱스 매핑
FDI_CLASS_MAPPING = {
    11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5,
    21: 6, 22: 7, 23: 8, 24: 9, 25: 10, 26: 11,
    31: 12, 32: 13, 33: 14, 34: 15, 35: 16, 36: 17,
    41: 18, 42: 19, 43: 20, 44: 21, 45: 22, 46: 23
}

# 클래스 인덱스를 FDI 번호로 변환하는 역매핑 생성
REVERSE_FDI_MAPPING = {v: k for k, v in FDI_CLASS_MAPPING.items()}

def point_in_polygon(point, polygon):
    """
    한 점이 다각형 내부에 있는지 확인하는 함수
    """
    x, y = point
    inside = False
    
    j = len(polygon) - 1
    for i in range(len(polygon)):
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and
            x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
        j = i
    
    return inside

def match_diseases_to_teeth(detections, segmentations):
    """
    치아 질환 데이터와 치아 경계 데이터를 매칭하여 각 치아별 질환을 찾아내는 함수
    """
    tooth_diseases = {}
    
    # detection 데이터 형식 변환
    formatted_detections = []
    for det in detections:
        # 문자열을 딕셔너리로 변환
        if isinstance(det, str):
            values = det.split()
            formatted_detections.append({
                "class_id": int(values[0]),
                "points": [float(values[1]), float(values[2]), 
                          float(values[3]), float(values[4])],
                "conf": float(values[5])
            })
        else:
            formatted_detections.append(det)
    
    # segmentation 데이터 형식 변환
    formatted_segments = []
    for seg in segmentations:
        if isinstance(seg, str):
            # 문자열을 좌표 리스트로 변환
            values = seg.split()
            class_id = int(values[0])
            points = []
            for i in range(1, len(values), 2):
                points.append([float(values[i]), float(values[i+1])])
            formatted_segments.append({
                "class_id": class_id,
                "points": points
            })
        else:
            formatted_segments.append(seg)
    
    t=0
    # 각 치아 세그멘테이션에 대해
    for seg in formatted_segments:
        class_idx = seg["class_id"]
        tooth_number = REVERSE_FDI_MAPPING[class_idx]
        tooth_region = seg["points"]
        
        diseases = []
        
        # 각 질환 탐지 결과에 대해
        for detection in formatted_detections:
            # detection box의 중심점 계산
            center_x = detection["points"][0] #+ detection["points"][2]/2
            center_y = detection["points"][1] #+ detection["points"][3]/2
            
            # detection box의 중심점이 치아 영역 내에 있는지 확인
            if point_in_polygon((center_x, center_y), tooth_region):
                disease_info = {
                    "disease_id": detection["class_id"],
                    "confidence": detection["conf"],
                    "location": detection["points"]
                }
                diseases.append(disease_info)
        
        if diseases:
            tooth_diseases[tooth_number] = diseases
    
    return tooth_diseases

