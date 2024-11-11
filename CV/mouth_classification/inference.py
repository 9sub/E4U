import torch
from torchvision import transforms
from PIL import Image
import argparse

# 데이터 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지 크기 조정
    transforms.ToTensor(),          # Tensor로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 이미지 정규화
])

# 모델 로드 함수
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)  # 전체 모델을 로드
    model.to(device)
    model.eval()
    return model

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 이미지 열기 및 RGB 변환
    return transform(image).unsqueeze(0)  # 배치 차원 추가하여 (1, C, H, W) 형태로 변환

# 추론 함수
def infer(model, image_path, device):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# main 함수: argparse 사용
def main():
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument('--model_path', type=str, default='./output_model/mobilenetv2_epoch20_lr1e-5_batch16.pth', help="Path to the saved model file (e.g., ./output_model/best_model.pth)")
    parser.add_argument('--image_path', type=str, default='/Users/igyuseob/Downloads/image1.jpg', help="Path to the image file for inference")
    args = parser.parse_args()

    # MPS 또는 CPU 선택
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 로드 및 추론 수행
    model = load_model(args.model_path, device)
    predicted_class = infer(model, args.image_path, device)

    # 결과 출력
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()