import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import argparse
import wandb
import numpy as np

# Seed 설정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 커스텀 Dataset 클래스
class OralImageDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        label = self.data[idx]['label']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지 크기 조정 (512x512)
    transforms.ToTensor(),          # Tensor로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 이미지 정규화
])

# 모델 정의 (EfficientNet 사용)
def get_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 구강, 비구강 이진 분류
    return model

# 모델 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, save_path='./output_model/best_model.pth'):
    model.train()
    model.to(device)
    best_val_loss = float('inf')  # 초기 베스트 검증 손실을 무한대로 설정
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        
        # Training loop
        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_loader_tqdm.set_postfix(loss=running_loss / (total / len(images)), accuracy=100 * correct / total)
        
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # 검증 데이터 평가
        val_accuracy, val_loss, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)
        
        # 베스트 모델 저장 (loss 기준)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, save_path)  # 모델 전체를 저장
            print(f"Best model saved with val_loss: {val_loss:.4f}")

        # wandb에 로그 기록
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# 모델 평가 함수 (loss 및 평가 지표 포함)
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    # tqdm을 사용하여 검증 데이터 평가
    data_loader_tqdm = tqdm(data_loader, desc='Validation')
    with torch.no_grad():
        for images, labels in data_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # tqdm 진행 막대 업데이트
            data_loader_tqdm.set_postfix(val_loss=running_loss / (total / len(images)), val_accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    loss = running_loss / len(data_loader)
    
    # 평가 지표 계산
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return accuracy, loss, precision, recall, f1

# main 함수: argparse 사용
def main():
    parser = argparse.ArgumentParser(description='구강 및 비구강 이미지 분류 모델 학습')
    parser.add_argument('--json_file', type=str, required=True, help='라벨 정보가 담긴 JSON 파일 경로')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--project_name', type=str, default="oral-image-classification", help='wandb 프로젝트 이름')
    parser.add_argument('--seed', type=int, default=42, help='seed 값 설정')
    parser.add_argument('--save_path', type=str, default='./output_model', help='모델 저장 경로')

    args = parser.parse_args()

    # Seed 설정
    set_seed(args.seed)

    # MPS 또는 CPU 선택
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # wandb 초기화
    wandb.init(project='capstone')
    wandb.run.name = args.project_name

    # 데이터셋 로드
    dataset = OralImageDataset(args.json_file, transform=transform)

    # 데이터셋 나누기 (0.8:0.1:0.1 비율)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 모델 생성
    model = get_model()

    # 손실 함수 및 최적화 함수 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    save_model_path = args.save_path + '/' + args.project_name + '.pth'

    # 학습 실행
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.epochs, save_path=save_model_path)

    # 테스트 데이터에서 모델 성능 평가
    test_accuracy, test_loss, precision, recall, f1 = evaluate_model(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # wandb에 테스트 결과 기록
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    main()