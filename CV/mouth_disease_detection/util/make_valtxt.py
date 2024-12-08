import os

#파일이 저장된 폴더 경로
folder_path = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/images/val"
output_file = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/val.txt"

#절대 경로 저장
with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 파일의 절대 경로 작성
            absolute_path = os.path.join(root, file)
            outfile.write(absolute_path + "\n")

print(f"모든 파일의 절대 경로가 '{output_file}'에 저장되었습니다.")