import numpy as np
import cv2

def calculate_iou(bbox_polygon, seg_mask):
    """
    바운딩 박스와 세그먼트 폴리곤 간의 IoU 계산 함수.
    
    Parameters:
        bbox_polygon (np.ndarray): 바운딩 박스 폴리곤 (N, 2).
        seg_mask (np.ndarray): 세그먼트 마스크 (이미지 크기와 동일한 크기).
        
    Returns:
        float: IoU (Intersection over Union) 값.
    """
    # 바운딩 박스 폴리곤 마스크 생성
    bbox_mask = np.zeros_like(seg_mask, dtype=np.uint8)
    cv2.fillPoly(bbox_mask, [bbox_polygon], 1)

    # Intersection (교집합) 계산
    intersection_mask = cv2.bitwise_and(bbox_mask, seg_mask)
    intersection_area = np.sum(intersection_mask)

    # 바운딩 박스 넓이와 세그먼트 넓이 계산
    bbox_area = np.sum(bbox_mask)
    seg_area = np.sum(seg_mask)

    # Union (합집합) 계산
    union_area = bbox_area + seg_area - intersection_area

    # IoU 계산
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou



def detect_teeth(user_status):
    detected_teeth = []
    threshold_iou = 0.7  # IoU 기준값

    for bbox_item in user_status.bounding_box:
        bbox = bbox_item['points']  # [x, y, w, h] 형식
        class_id = bbox_item['class_id']
        #print(class_id)

        for seg_item in user_status.segmentation_data:
            points = seg_item['points']
            if len(points) < 4:
                print("재구성할 수 있는 충분한 점이 없습니다:", points)
                continue  # 이 반복을 건너뛰    
            # Segmentation points를 numpy 배열로 변환하고 폴리곤 생성
            points = np.array(points).reshape(-1, 2).astype(np.int32)
            mask = np.zeros((user_status.image_size[0], user_status.image_size[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1 ) 
            # Detection 바운딩 박스를 [x1, y1, x2, y2] 형식으로 변환
            x, y, w, h = bbox
            bbox_polygon = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)    
            # IoU 계산
            iou = calculate_iou(bbox_polygon, mask)
            #print(f"Class ID: {class_id}, IoU: {iou:.4f}"  
            if iou >= threshold_iou:
                detected_teeth.append(class_id)
            print('----------------------', detected_teeth)

    return list(set(detected_teeth))  # 중복 제거