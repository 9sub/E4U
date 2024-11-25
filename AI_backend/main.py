from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
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
from schema import InputText, GPTRequest, UserStatus, result_report
from gpt import call_gpt
from utils.calculate_max_tokens import check_max_tokens
from utils.read_bounding_box import find_bounding_box
from utils.read_segmentation import read_segmentation_file
from utils.check_image_size import check_image_size
from utils.check_disease_teeth_num import match_diseases_to_teeth_and_gums
from utils.remove_dup import remove_dup
from utils.return_json_format import return_json_format
from danger_point import calculate_danger_score
from utils.analysis_results_form import analysis_results_form
from utils.result_report_form import result_report_form

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
        device = torch.device("mps" if torch.backends.mps.is_available else "cpu")

        # 모델 로드
        model_path = './models/mobilenetv2_epoch20_lr1e-5_batch16.pth'
        model = load_model(model_path, device)
        # 추론 수행
        predicted_class = infer(model, image_tensor, device)

        is_mouth_image = (predicted_class == 1)
        print(is_mouth_image)
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
    print(user_status.result)
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
    json_format = return_json_format(user_status.result)
    return json_format

# @app.post("/mouth_front_left_right/")
# async def detect_mouth_front_left_right(file: UploadFile = File(...)):
#     temp_file = f"./result/mouth_front_left_right/before_inference/{uuid.uuid4()}.jpg"
#     with open(temp_file, 'wb') as buffer:
#         shutil.copyfileobj(file.file, buffer)


@app.post('/danger_point/')
async def get_danger_point(pain_level: int):
    user_status.pain_level = pain_level
    danger_point=calculate_danger_score(user_status.result, user_status.pain_level)
    return {"danger_point": danger_point}
    


# @app.post("/et5_predict/")
# async def predict(input_data: InputText):
#     input_text = input_data.text
#     result = generate_answer(input_text)
#     return {"output": result}


# @app.get('/get_painscore')
# async def get_painscore(level: int):
#     user_status.pain_level = level
#     return {"pain_level": level, "message": f"Received pain level: {level}"}


@app.post('/result_report/')
async def result_report(data : result_report):
    input_text_result, input_text_detailed_result = result_report_form(data)

    input_text_result += "\n 질병 위치와 치과방문 권유만 분석해. 하나의 텍스트로 작성해. 말투는 정중하게 마지막 인사는 제외해."
    input_text_detailed_result += "\n 환자의 증상과 질환을 통해 원인과 증상만 자세하게 분석하고 치아위치는 제외해. 하나의 텍스트로 작성해. 말투는 정중하게 사용자에게 말하는 것처럼, 마지막 인사는 제외해."
    input_text_care_method = input_text_result + "\n 발생한 질병을 관리할 수 있는 방법을 도구와 관리팁으로만 작성해. 치아위치나 치과 방문 이야기는 제외해. 하나의 텍스트로 작성해. 말투는 정중하게."
    result = call_gpt(prompt=input_text_result)
    detailed_result = call_gpt(prompt=input_text_detailed_result)
    care_method = call_gpt(prompt=input_text_care_method)
    
    return {"result": result, "detailed_result": detailed_result, "care_method" : care_method}

@app.post('/chat/')
async def chat(input_data: InputText):
    input_text = input_data.text
    result = call_gpt(prompt=input_text)
    
    return {"output": result}

'''
{
  "text": "(tooth_num 11 : 충치 tooth_num 31 : 충치 , 위와 아래 치아가 너무 시린 증상이 있어) 괄호 안에는 각 치아별로 질환이 적혀져 있고 쉼표 뒤에는 환자의 증상이 있어 너가 구강질환에 대한 진단을 해줘"
}
'''


# @app.post('/analysis_results/')
# async def analysis_results():
#     result_text = analysis_results_form(user_status.result)
#     return {"output": result_text}


# import base64

# def encode_image_to_base64(file_path: str) -> str:
#     """
#     로컬 이미지 파일을 Base64로 인코딩
#     """
#     with open(file_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

# @app.post("/gpt/")
# async def gpt_endpoint(prompt: str=Form(...), file: UploadFile = File(...)):
#     temp_file = f"./result/gpt_image/{uuid.uuid4()}.jpg"
#     with open(temp_file, 'wb') as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     image_base64 = encode_image_to_base64(temp_file)

#     max_tokens=check_max_tokens(prompt)
#     result = call_gpt(prompt=prompt,image_file=image_base64, max_tokens=300)
#     return {"response": result, "max_tokens": 500}