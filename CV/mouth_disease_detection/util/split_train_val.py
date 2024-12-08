import os
import random
import shutil

# 경로 설정
image_dir = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/images/train"
label_dir = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/labels/train"
val_image_dir = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/images/val"
val_label_dir = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/labels/val"
train_txt = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/train.txt"

# 디렉토리 생성
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

# 검증 데이터 비율 설정 (예: 10%)
val_split_ratio = 0.1
val_count = int(len(image_files) * val_split_ratio)

# 랜덤하게 이미지 파일 선택
val_files = random.sample(image_files, val_count)

# 검증 데이터셋으로 옮겨진 파일 경로 리스트
val_file_paths = []

# 파일 이동
for image_file in val_files:
    base_name = os.path.splitext(image_file)[0]
    label_file = f"{base_name}.txt"

    # 파일 경로 설정
    src_image_path = os.path.join(image_dir, image_file)
    src_label_path = os.path.join(label_dir, label_file)
    dst_image_path = os.path.join(val_image_dir, image_file)
    dst_label_path = os.path.join(val_label_dir, label_file)

    # 이미지와 라벨 파일 이동
    if os.path.exists(src_image_path) and os.path.exists(src_label_path):
        shutil.move(src_image_path, dst_image_path)
        shutil.move(src_label_path, dst_label_path)
        print(f"Moved: {image_file} and {label_file}")
        # 옮겨진 이미지 파일의 절대 경로 추가
        val_file_paths.append(os.path.join(image_dir, image_file))

# train.txt 수정
with open(train_txt, "r", encoding="utf-8") as infile:
    train_lines = infile.readlines()

# 옮겨진 파일 경로 제거
with open(train_txt, "w", encoding="utf-8") as outfile:
    for line in train_lines:
        if not any(val_file in line for val_file in val_file_paths):
            outfile.write(line)

print("검증 데이터셋 생성 및 train.txt 업데이트 완료!")