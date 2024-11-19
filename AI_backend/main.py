from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from io import BytesIO
import torch
from torchvision.models.mobilenetv2 import MobileNetV2
import torch
from ultralytics import YOLO
import uuid
import os
import shutil
import time
from fastapi.middleware.cors import CORSMiddleware

from predict.is_mouth_predict import preprocess_image, infer, load_model
from predict.et5_predict import generate_answer
from schema import InputText, GPTRequest, UserStatus
from gpt import call_gpt
from utils.calculate_max_tokens import check_max_tokens
from utils.read_bounding_box import find_bounding_box
from utils.read_segmentation import read_segmentation_file
from utils.check_image_size import check_image_size
from utils.check_disease_teeth_num import match_diseases_to_teeth_and_gums
from utils.remove_dup import remove_dup

app = FastAPI()

user_status = UserStatus()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Model server API"}



@app.post("/is_mouth/")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 읽기 및 전처리
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        

        # MPS 또는 CPU 선택
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 로드
        model_path = './models/mobilenetv2_epoch20_lr1e-5_batch16.pth'
        model = load_model(model_path, device)
        # 추론 수행
        predicted_class = infer(model, image_tensor, device)

        is_mouth_image = (predicted_class == 1)

        return {"is_mouth_image": is_mouth_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/")
def detect_image(file: UploadFile = File(...)):
    temp_file = f"./result/before_inference/{uuid.uuid4()}.jpg"
    with open(temp_file, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    user_status.image_size = check_image_size(temp_file)

    # detection 모델 로드 및 추론
    detection_model = YOLO('./models/detection_final.pt')
    detection_results = detection_model.predict(temp_file, conf=0.2, save=True, project="result", name="detection", exist_ok=True)

    # 결과 파일 경로 설정 및 확인
    detection_output_image_path = f"result/detection/{os.path.basename(temp_file)}"
    before_detection_image_path = f"result/before_inference/{os.path.basename(temp_file)}"

    #detection 결과 이미지 path 저장, bounding box 저장
    user_status.detection_image_path = detection_output_image_path
    user_status.bounding_box=find_bounding_box(before_detection_image_path, detection_results)

    #segmentation 모델 로드 및 추론
    segmentation_model = YOLO('./models/segmentation_last.pt')
    segmentation_model.predict(temp_file, save=True, project="result", name="segmentation", save_txt=True, exist_ok=True)

    # 결과 파일 경로 설정 및 확인
    segmentation_output_image_path = f"result/segmentation/{os.path.basename(temp_file)}"

    #segmentation 결과 이미지 path 저장
    user_status.segmentation_image_path = segmentation_output_image_path

    txt_filename = os.path.splitext(os.path.basename(temp_file))[0] + ".txt"
    txt_file_path = os.path.join("result/segmentation/labels", txt_filename)
    
    user_status.segmentation_image_path = segmentation_output_image_path
    user_status.segmentation_data = read_segmentation_file(txt_file_path)

    #print(user_status.bounding_box)

    #print(user_status.segmentation_data)
    result = match_diseases_to_teeth_and_gums(user_status.bounding_box, user_status.segmentation_data, user_status.image_size[0], user_status.image_size[1])
    #중복 제거
    result = remove_dup(result)

    user_status.result = result

    # 출력 로직
    print("===== 치아 질환 =====")
    for tooth_num, diseases in result['tooth_diseases'].items():
        print(f"치아 {tooth_num}번의 질환:")
        for disease in diseases:
            print(f"  - 질환 ID: {disease['disease_id']}")
            print(f"    질환 이름: {disease['disease_name']}")
            print(f"    신뢰도: {disease['confidence']:.2f}")
            print(f"    위치: {disease['location']}")

    print("\n===== 잇몸 질환 =====")
    for region, diseases in result['gum_diseases'].items():
        print(f"잇몸 부위: {region}")
        for disease in diseases:
            print(f"  - 질환 ID: {disease['disease_id']}")
            print(f"    질환 이름: {disease['disease_name']}")
            print(f"    신뢰도: {disease['confidence']:.2f}")
            print(f"    위치: {disease['location']}")

    # 결과 파일이 생성될 때까지 최대 30초 대기
    timeout = 30
    while not os.path.exists(detection_output_image_path) and os.path.exists(segmentation_output_image_path) and timeout > 0:
        time.sleep(1)
        timeout -= 1
    if not os.path.exists(detection_output_image_path) and os.path.exists(segmentation_output_image_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(detection_output_image_path, media_type="image/jpg")


@app.post("/detection_result/")
def get_detection_result():
    return user_status.result

@app.post("/mouth_front_left_right/")
async def detect_mouth_front_left_right(file: UploadFile = File(...)):
    temp_file = f"./result/mouth_front_left_right/before_inference/{uuid.uuid4()}.jpg"
    with open(temp_file, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)


@app.post('/danger_point/')
async def get_danger_point():
    return {"danger_point": 1}
    


@app.post("/et5_predict/")
async def predict(input_data: InputText):
    input_text = input_data.text
    result = generate_answer(input_text)
    to_gpt = "아래 내용 중 의심되는 구강질환을 말해. 그리고 내용 요약하면서 문맥을 매끄럽게 만들어." + f"<{result}>"
    max_tokens=check_max_tokens(to_gpt)
    result = call_gpt(prompt=to_gpt, max_tokens=max_tokens)
    user_status.chating = {"input": input_text, "output": result}
    print(user_status.chating)
    return {"output": result}


@app.get('/get_painscore')
async def get_painscore(level: int):
    user_status.pain_level = level
    return {"pain_level": level, "message": f"Received pain level: {level}"}


# @app.post("/gpt/")
# async def gpt_endpoint(request: GPTRequest):
#     max_tokens=check_max_tokens(request.prompt)
#     result = call_gpt(prompt=request.prompt, max_tokens=max_tokens)
#     return {"response": result, "max_tokens": max_tokens}