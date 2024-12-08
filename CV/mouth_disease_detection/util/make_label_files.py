import os

# 이미지와 라벨 디렉토리 경로
image_dir = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/images/train"
label_dir = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/labels/train"

# 이미지 파일 목록 가져오기 (JPG, PNG, JPEG 확장자 포함)
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

# 빈 라벨 파일 생성
for image_file in image_files:
    # 이미지 파일명에서 확장자 제거 후 .txt 파일 이름 생성
    base_name = os.path.splitext(image_file)[0]
    label_file = f"{base_name}.txt"
    label_path = os.path.join(label_dir, label_file)

    # 라벨 파일이 없는 경우 빈 파일 생성
    if not os.path.exists(label_path):
        with open(label_path, "w") as file:
            pass  # 빈 파일 생성
        print(f"빈 라벨 파일 생성: {label_path}")
    else:
        print(f"라벨 파일 존재: {label_path} (패스)")

print("작업이 완료되었습니다.")