import os

def read_segmentation_file(txt_file_path: str):
    segmentation_data = []

    # 파일이 존재하는지 확인
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {txt_file_path}")

    with open(txt_file_path, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # 데이터 파싱
            values = line.split()
            class_id = int(values[0])
            coordinates = list(map(float, values[1:]))

            # (x, y) 좌표 쌍으로 변환
            points = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]

            # 객체 데이터를 리스트에 추가
            segmentation_data.append({
                "class_id": class_id,
                "points": points
            })
    return segmentation_data