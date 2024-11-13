import os

# 입력 및 출력 파일 경로
input_file = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/before_change_train.txt"
output_file = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/train.txt"

# 절대 경로 설정
base_path = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/images/train"

# 파일 읽기 및 경로 수정
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 파일명 추출 및 절대 경로로 변경
        filename = line.strip().split("/")[-1]
        absolute_path = os.path.join(base_path, filename)
        outfile.write(absolute_path + "\n")

print(f"경로가 수정된 파일이 '{output_file}'에 저장되었습니다.")