from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
import torch
from torchvision.models.mobilenetv2 import MobileNetV2
import torch

torch.serialization.add_safe_globals([MobileNetV2])



from is_mouth_predict import preprocess_image, infer, load_model
from et5_predict import generate_answer
from schema import InputText, GPTRequest
from gpt import call_gpt

app = FastAPI()


@app.post("/et5-predict")
async def predict(input_data: InputText):
    input_text = input_data.text
    result = generate_answer(input_text)
    return {"output": result}

@app.post("/predict/")
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
        is_oral_image = (predicted_class == 1)

        return {"is_oral_image": is_oral_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gpt/")
async def gpt_endpoint(request: GPTRequest):
    result = call_gpt(prompt=request.prompt, max_tokens=request.max_tokens)
    return {"response": result}