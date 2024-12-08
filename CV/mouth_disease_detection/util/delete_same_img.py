import os

# 파일 경로 설정
train_txt = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/train.txt"
val_txt = "/Users/igyuseob/Desktop/capstone/data/구강질환_detection/dataset_all/data/val.txt"

# `train.txt`에서 중복 제거
with open(train_txt, "r", encoding="utf-8") as file:
    train_paths = list(set(line.strip() for line in file))

# `val.txt`에서 중복 제거
with open(val_txt, "r", encoding="utf-8") as file:
    val_paths = list(set(line.strip() for line in file))

# 중복 제거된 내용을 다시 파일에 저장
with open(train_txt, "w", encoding="utf-8") as file:
    for path in sorted(train_paths):
        file.write(path + "\n")

with open(val_txt, "w", encoding="utf-8") as file:
    for path in sorted(val_paths):
        file.write(path + "\n")

print("각각의 파일에서 중복된 항목이 제거되었습니다.")