import os
import cv2
from utils.nms import nms

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


def find_bounding_box(image_path: str, results):
    txt_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    txt_file_path = os.path.join("result/detection/labels", txt_filename)
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)

    # 모든 검출 결과를 저장할 리스트
    detections = []

    # YOLO의 결과에서 박스를 추출
    for result in results:
        boxes = result.boxes

        # 각 박스에서 필요한 정보 추출
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())  # 클래스 ID
            x, y, w, h = boxes.xywhn[i].tolist()  # Bounding Box 좌표 (x_center, y_center, width, height)
            conf = float(boxes.conf[i].item())  # 신뢰도 (confidence score)

            # 검출 결과 추가: [class_id, x, y, w, h, conf]
            detections.append([class_id, x, y, w, h, conf])

    suppressed_boxes = nms(detections, threshold=0.6)

    image = cv2.imread(image_path)

    bbox = []

    with open(txt_file_path, "w") as file:
        for box in suppressed_boxes:
            class_id, x, y, w, h, conf = box
            file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
            bbox.append({
                'class_id': class_id,
                'points': [x, y, w, h],
                'conf': conf
            })

            # 시각화: 박스 그리기
            x_min = int((x - w / 2) * image.shape[1])
            y_min = int((y - h / 2) * image.shape[0])
            x_max = int((x + w / 2) * image.shape[1])
            y_max = int((y + h / 2) * image.shape[0])


            class_name = class_map.get(class_id)
            # 박스 그리기 (클래스 ID 및 신뢰도 표시)
            color = (0, 255, 0)  # 녹색
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    output_image_path = f"result/detection/{os.path.basename(image_path)}"
    cv2.imwrite(output_image_path, image)
    print(f"시각화된 결과 이미지가 저장되었습니다: {output_image_path}")

    return bbox