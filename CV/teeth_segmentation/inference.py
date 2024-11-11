import mmcv
from mmdet.apis import init_detector, inference_detector
import torch
from mmdet_custom.models.backbones.vit_adapter import ViTAdapter
from mmcv.runner import load_checkpoint
import os, json
import cv2
import numpy as np

# Config 파일과 체크포인트 경로 설정, device cuda 설정
config_file = '/workspace/AI/teeth_segmentation/configs/mask_rcnn/mask_rcnn_deit_adapter_small_fpn_3x_coco_shk.py'
checkpoint_file = '/workspace/AI/teeth_segmentation/work_dirs/mask_rcnn_deit_adapter_small_fpn_3x_coco_shk/latest.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 모델 초기화
model = init_detector(config_file, checkpoint_file, device=device)
load_checkpoint(model, checkpoint_file, map_location=device, strict=False)

# 테스트 이미지 경로
image_path = '/workspace/ViT-Adapter/detection/data/NIA/4/dest_images/right_20417.png'
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

#segmentation 결과값
result = inference_detector(model, image_path)

# # 클래스 ID와 치아 번호 매핑
# tooth_number_mapping = [
#     "Tooth 11", "Tooth 12", "Tooth 13", "Tooth 14", "Tooth 15", "Tooth 16", #왼쪽 위 치아
#     "Tooth 21", "Tooth 22", "Tooth 23", "Tooth 24", "Tooth 25", "Tooth 26", #오른쪽 위 치아
#     "Tooth 31", "Tooth 32", "Tooth 33", "Tooth 34", "Tooth 35", "Tooth 36", #오른쪽 아래 치아
#     "Tooth 41", "Tooth 42", "Tooth 43", "Tooth 44", "Tooth 45", "Tooth 46", #왼쪽 아래 치아
# ]

# FDI 번호 할당 함수
def assign_fdi_number(x_center, y_center):
    if y_center < horizontal_center:  # 상악 (위쪽)
        if x_center < vertical_center:
            return "Tooth 21"  # 좌측 상악
        else:
            return "Tooth 11"  # 우측 상악
    else:  # 하악 (아래쪽)
        if x_center < vertical_center:
            return "Tooth 31"  # 좌측 하악
        else:
            return "Tooth 41"  # 우측 하악


bbox_results, mask_results = result

# image 불러오기
image = cv2.imread(image_path)
# image 높이, 너비 측정
image_height, image_width = image.shape[:2]
image = image.copy()

# 중앙선 기준 계산
vertical_center = image_width // 2
horizontal_center = image_height // 2

bboxes, masks = result
object_count=0

# FDI 번호 리스트 초기화
upper_left = ["Tooth 21", "Tooth 22", "Tooth 23", "Tooth 24", "Tooth 25", "Tooth 26", "Tooth 27", "Tooth 28"]
upper_right = ["Tooth 11", "Tooth 12", "Tooth 13", "Tooth 14", "Tooth 15", "Tooth 16", "Tooth 17", "Tooth 18"]
lower_left = ["Tooth 31", "Tooth 32", "Tooth 33", "Tooth 34", "Tooth 35", "Tooth 36", "Tooth 37", "Tooth 38"]
lower_right = ["Tooth 41", "Tooth 42", "Tooth 43", "Tooth 44", "Tooth 45", "Tooth 46", "Tooth 47", "Tooth 48"]

# 치아 위치별 저장 리스트
upper_left_teeth = []
upper_right_teeth = []
lower_left_teeth = []
lower_right_teeth = []

# 중심 좌표 계산 및 치아 분류
for (bboxes, masks) in zip(bbox_results, mask_results):
    for (bbox, mask) in zip(bboxes, masks):
        if len(bbox) == 5:
            x_min, y_min, x_max, y_max, score = bbox
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

            # 신뢰도 점수 필터링
            if score < 0.3:
                continue

            # 치아 위치별 분류
            if y_center < horizontal_center:  # 상악 (위쪽)
                if x_center < vertical_center:
                    upper_left_teeth.append((bbox, mask))
                else:
                    upper_right_teeth.append((bbox, mask))
            else:  # 하악 (아래쪽)
                if x_center < vertical_center:
                    lower_left_teeth.append((bbox, mask))
                else:
                    lower_right_teeth.append((bbox, mask))

# 가운데를 기준으로 FDI 치아 번호 할당
def assign_teeth_numbers(teeth_list, fdi_numbers, image):
    # 중앙에 가까운 순서로 정렬
    teeth_list.sort(key=lambda item: abs((item[0][0] + item[0][2]) // 2 - vertical_center))
    for idx, (bbox, mask) in enumerate(teeth_list):
        if idx >= len(fdi_numbers):
            break  # 할당할 FDI 번호가 없으면 종료

        x_min, y_min, x_max, y_max, score = bbox
        tooth_number = fdi_numbers[idx]

        # Bounding Box 그리기
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

        # 마스크 그리기
        if np.any(mask):
            mask = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, thickness)

        # 텍스트 표시 (FDI 번호 및 신뢰도 점수)
        text = f"{tooth_number} ({score:.2f})"
        cv2.putText(image, text, (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

assign_teeth_numbers(upper_left_teeth, upper_left, image)
assign_teeth_numbers(upper_right_teeth, upper_right, image)
assign_teeth_numbers(lower_left_teeth, lower_left, image)
assign_teeth_numbers(lower_right_teeth, lower_right, image)

# 결과 이미지 저장
output_image_path_num = os.path.join(output_dir, 'teethnum.png')
cv2.imwrite(output_image_path_num, image)
# 결과 이미지 저장

print(f"결과가 저장되었습니다: {output_image_path_num}")