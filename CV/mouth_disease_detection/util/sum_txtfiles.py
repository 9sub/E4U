import os
import random

# 입력 텍스트 파일들이 있는 디렉토리 경로
input_dir = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/txtfiles"
output_file = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/train.txt"

# 모든 줄을 저장할 리스트
all_lines = []

# 입력 디렉토리에서 모든 텍스트 파일 읽기
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            all_lines.extend(lines)

# 모든 줄을 랜덤하게 섞기
random.shuffle(all_lines)

# 결과를 새로운 파일에 저장
with open(output_file, "w", encoding="utf-8") as output:
    output.writelines(all_lines)

print(f"파일이 '{output_file}'에 성공적으로 저장되었습니다.")