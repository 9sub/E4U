# FDI 번호와 클래스 인덱스 매핑
FDI_CLASS_MAPPING = {
    11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5,
    21: 6, 22: 7, 23: 8, 24: 9, 25: 10, 26: 11,
    31: 12, 32: 13, 33: 14, 34: 15, 35: 16, 36: 17,
    41: 18, 42: 19, 43: 20, 44: 21, 45: 22, 46: 23
}

# 클래스 인덱스를 FDI 번호로 변환하는 역매핑
REVERSE_FDI_MAPPING = {v: k for k, v in FDI_CLASS_MAPPING.items()}

# 질병 클래스 매핑
DISEASE_CLASS_MAP = {
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

# 잇몸 질환 리스트
GUM_DISEASES = ['CaS', 'CoS', 'Gingivitis', 'Gum', 'MC', 'MouthUlcer']

# 잇몸 영역 정의 (이미지 크기에 따라 조정 필요)
GUM_REGIONS = {
    '상악_좌측후방': {'x': (0, 0.25), 'y': (0, 0.5)},
    '상악_좌측중간': {'x': (0.25, 0.5), 'y': (0, 0.5)},
    '상악_우측중간': {'x': (0.5, 0.75), 'y': (0, 0.5)},
    '상악_우측후방': {'x': (0.75, 1.0), 'y': (0, 0.5)},
    '하악_좌측후방': {'x': (0, 0.25), 'y': (0.5, 1.0)},
    '하악_좌측중간': {'x': (0.25, 0.5), 'y': (0.5, 1.0)},
    '하악_우측중간': {'x': (0.5, 0.75), 'y': (0.5, 1.0)},
    '하악_우측후방': {'x': (0.75, 1.0), 'y': (0.5, 1.0)}
}

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

def get_gum_region(x, y, img_width, img_height):
    """
    좌표가 어느 잇몸 영역에 속하는지 확인하는 함수
    """
    normalized_x = x# / img_width
    normalized_y = y# / img_height
    
    for region_name, bounds in GUM_REGIONS.items():
        if (bounds['x'][0] <= normalized_x <= bounds['x'][1] and
            bounds['y'][0] <= normalized_y <= bounds['y'][1]):
            return region_name
    return None

def match_diseases_to_teeth_and_gums(detections, segmentations, img_width, img_height):
    """
    치아 질환 데이터와 치아 경계 데이터를 매칭하여 각 치아별, 잇몸 영역별 질환을 찾아내는 함수
    """
    tooth_diseases = {}
    gum_diseases = {region: [] for region in GUM_REGIONS.keys()}
    etc_diseases = {}

    # detection 데이터 형식 변환
    formatted_detections = []
    for det in detections:
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

    # 각 detection에 대해 처리
    before_tooth_num =-1
    for detection in formatted_detections:
        disease_name = DISEASE_CLASS_MAP[detection["class_id"]]
        
        # detection box의 중심점 계산
        center_x = detection["points"][0]
        center_y = detection["points"][1]
        
        disease_info = {
            "disease_id": detection["class_id"],
            "disease_name": disease_name,
            "confidence": detection["conf"],
            "location": detection["points"]
        }


        #etc
        if disease_name in ['CaS', 'CoS','OLP']:
            #print("etc:",disease_name)
            region = "혀" if disease_name in ['CaS', 'OLP'] else "입술"
            if region not in etc_diseases:
                etc_diseases[region] = []
            etc_diseases[region].append({
                "disease_id": detection["class_id"],
                "disease_name": disease_name,
                "confidence": detection["conf"],
                "location": detection["points"] 
            })
        
        # 잇몸 질환인 경우
        elif disease_name in GUM_DISEASES:
            #print("gum:",disease_name)
            region = get_gum_region(center_x, center_y, img_width, img_height)
            if region:
                gum_diseases[region].append(disease_info)
        
        # 치아 질환인 경우
        else:
            # 각 치아 세그멘테이션과 매칭
            #print("tooth:",disease_name)
            if before_tooth_num == -1:
                tooth_number = REVERSE_FDI_MAPPING[8]
            else:
                tooth_number = before_tooth_num
            tmp=0
            for seg in formatted_segments:
                if disease_name == "Hypodontia" and tmp==0:
                        tooth_diseases[tooth_number].append(disease_info)
                        tmp+=1
                        
                else:
                    if point_in_polygon((center_x, center_y), seg["points"]):
                        tooth_number = REVERSE_FDI_MAPPING[seg["class_id"]]     
                        before_tooth_num = tooth_number
                        if tooth_number not in tooth_diseases:
                            tooth_diseases[tooth_number] = []
                        tooth_diseases[tooth_number].append(disease_info)


    #print(tooth_diseases)
    return {
        "tooth_diseases": tooth_diseases,
        "gum_diseases": gum_diseases,
        "etc" : etc_diseases
    }