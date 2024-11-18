def calculate_iou(box1, box2):
    # box 형식: [x_center, y_center, width, height]
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # 겹치는 영역의 좌표 계산
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 겹치는 영역의 넓이 계산
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # 각 박스의 넓이 계산
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # IoU 계산
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def nms(detections, threshold=0.5):
    # 신뢰도 기준으로 정렬 (신뢰도가 마지막에 있음)
    detections = sorted(detections, key=lambda x: x[5], reverse=True)  
    suppressed = []

    while detections:
        current_box = detections.pop(0)  # 가장 신뢰도가 높은 박스를 선택
        suppressed.append(current_box)

        detections = [box for box in detections if calculate_iou(current_box[1:5], box[1:5]) < threshold]

    return suppressed

# 텍스트 파일 읽기
def read_detections_from_file(file_path):
    detections = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            detections.append(values)
    return detections
