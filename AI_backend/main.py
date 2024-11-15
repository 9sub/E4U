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


from predict.is_mouth_predict import preprocess_image, infer, load_model
from predict.et5_predict import generate_answer
from schema import InputText, GPTRequest
from gpt import call_gpt
from utils.calculate_max_tokens import check_max_tokens

app = FastAPI()

@app.post("/is_mouth/")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 읽기 및 전처리
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        

        # MPS 또는 CPU 선택
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        # 모델 로드
        model_path = '/Users/igyuseob/Desktop/ai_github/dev/AI_backend/models/mobilenetv2_epoch20_lr1e-5_batch16.pth'
        model = load_model(model_path, device)
        # 추론 수행
        predicted_class = infer(model, image_tensor, device)

        # 구강 이미지 여부 확인 (예시로 클래스 0이 구강 이미지로 가정)
        is_mouth_image = (predicted_class == 1)

        return {"is_mouth_image": is_mouth_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    temp_file = f"./result/before_inference/temp_{uuid.uuid4()}.jpg"
    with open(temp_file, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 모델 불러오기 및 추론
    model = YOLO('/Users/igyuseob/Desktop/ai_github/dev/AI_backend/models/detection_final.pt')
    model.predict(temp_file, conf=0.2, save=True, project="result", name="inference", exist_ok=True)

    # 결과 파일 경로 설정 및 확인
    output_image_path = f"result/inference/{os.path.basename(temp_file)}"
    # 결과 파일이 생성될 때까지 최대 30초 대기
    timeout = 30
    while not os.path.exists(output_image_path) and timeout > 0:
        time.sleep(1)
        timeout -= 1

    if not os.path.exists(output_image_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(output_image_path)


@app.post("/et5-predict")
async def predict(input_data: InputText):
    input_text = input_data.text
    result = generate_answer(input_text)
    to_gpt = "아래 내용 중 의심되는 구강질환을 말해. 그리고 내용 요약해." + f"<{result}>"
    max_tokens=check_max_tokens(to_gpt)
    result = call_gpt(prompt=to_gpt, max_tokens=max_tokens)
    return {"output": result}

# @app.post("/gpt/")
# async def gpt_endpoint(request: GPTRequest):
#     max_tokens=check_max_tokens(request.prompt)
#     result = call_gpt(prompt=request.prompt, max_tokens=max_tokens)
#     return {"response": result, "max_tokens": max_tokens}