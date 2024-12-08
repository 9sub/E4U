import os
import json
import argparse
import random

def main(args):

    # 이미지 경로와 라벨을 저장할 리스트 생성
    no_mouth_data = args.no_mouth_data
    mouth_data = args.mouth_data
    data = []

    # 비구강 이미지에는 라벨 0 할당
    for file_name in os.listdir(no_mouth_data):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):  # 파일 확장자에 맞게 조정
            file_path = os.path.join(no_mouth_data, file_name)
            data.append({"image_path": file_path, "label": 0})

    # 구강 이미지에는 라벨 1 할당
    for file_name in os.listdir(mouth_data):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):  # 파일 확장자에 맞게 조정
            file_path = os.path.join(mouth_data, file_name)
            data.append({"image_path": file_path, "label": 1})

    random.shuffle(data)

    # JSON 파일로 저장
    json_file_path = args.json_file_path
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print(f"라벨 JSON 파일이 {json_file_path}에 생성되었습니다.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='make json label file')

    parser.add_argument('--no_mouth_data', 
                        type=str, 
                        default='/Users/igyuseob/Desktop/capstone/data/VOCdevkit/VOC2012/JPEGImages', 
                        help='no mouth data path')
    
    parser.add_argument('--mouth_data',
                        type=str,
                        default='/Users/igyuseob/Desktop/capstone/data/mout_all_image',
                        help='mouth data path')
    
    parser.add_argument('--json_file_path',
                        type=str,
                        default='./image_labels.json',
                        help='output json file path')
    
    args = parser.parse_args()

    main(args)