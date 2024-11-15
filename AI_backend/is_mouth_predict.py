
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from torchvision.models.mobilenetv2 import MobileNetV2
import torch

torch.serialization.add_safe_globals([MobileNetV2])


# 데이터 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 모델 로드 함수
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model

# 이미지 전처리 함수
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# 추론 함수
def infer(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()