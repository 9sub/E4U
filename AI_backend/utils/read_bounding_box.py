import os

def find_bounding_box(image_path: str, results):
    txt_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    txt_file_path = os.path.join("result/detection/labels", txt_filename)
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)

    bbox = []

    with open(txt_file_path, "w") as file:
        for result in results:
            boxes = result.boxes

            # 각 Box에 대해 클래스 ID, Bounding Box, confidence 추출
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())  # 클래스 ID
                x, y, w, h = boxes.xywhn[i].tolist()  # Bounding Box 좌표 (x_center, y_center, width, height)
                conf = float(boxes.conf[i].item())  # 신뢰도 (confidence score)

                # .txt 파일에 기록
                file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

                # Bounding Box 정보 리스트에 추가
                bbox.append({'class_id':class_id, 
                             'points':[x, y, w, h], 
                             'conf':conf
                             })

    return bbox