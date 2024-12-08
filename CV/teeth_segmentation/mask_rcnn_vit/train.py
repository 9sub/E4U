import os
import json
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from PIL import ImageDraw
import numpy as np
from tqdm import tqdm

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

# Hyperparameters
BATCH_SIZE = 4
NUM_CLASSES = 18  # 치아 번호 1~32 + 배경
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# Custom Dataset
class ToothSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = []
        self.annotations = []
        self.transform = transform

        # 이미지와 라벨 파일 수집
        for subdir in ['1.right', '2.front', '3.left', '4.upper', '5.lower']:
            image_path = os.path.join(image_dir, subdir)
            label_path = os.path.join(label_dir, subdir)
            for label_file in os.listdir(label_path):
                if label_file.endswith(".json"):
                    with open(os.path.join(label_path, label_file), 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                        image_filepath = annotation["image_filepath"].replace("\\", "/")
                        image_name = os.path.basename(image_filepath)
                        self.image_files.append(os.path.join(image_path, image_name))
                        self.annotations.append(annotation["tooth"])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        annotation = self.annotations[idx]

        masks = []
        boxes = []
        labels = []

        for tooth in annotation:
            segmentation = tooth['segmentation']
            teeth_num = tooth['teeth_num']
            polygon = np.array(segmentation).reshape(-1, 2)
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon], outline=1, fill=1)
            mask = np.array(mask)
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            masks.append(mask)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(teeth_num)

        # numpy 배열로 변환 후 tensor로 변환
        masks = np.stack(masks)  # 리스트 대신 numpy 배열로 변환
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        # 이미지 변환만 적용
        if self.transform:
            image = self.transform(image)

        return image, target

# Data Augmentation
def get_transform(train):
    transforms = []
    transforms.append(T.Resize((256, 256)))  # 이미지 크기를 256x256으로 리사이즈
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# 모델 정의
def get_model_instance_segmentation(num_classes):
    # 사전 학습된 가중치 불러오기
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)

    # 분류 예측기 수정
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # 마스크 예측기 수정
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# 학습 함수
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    # tqdm 추가: 데이터 로더 반복 시 진행 바 표시
    for images, targets in tqdm(data_loader, desc="Training", leave=False):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Average Loss: {avg_loss:.4f}")


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        # 간단한 평가 출력 (첫 번째 출력만 표시)
        if len(outputs) > 0:
            print(outputs[0])


# 데이터 로드 및 모델 초기화
train_dataset = ToothSegmentationDataset(
    image_dir='/Users/igyuseob/Desktop/capstone/data/052.구강 이미지 합성데이터/052.구강 이미지 합성데이터/3.개방데이터/1.데이터/Training/1.원천데이터',
    label_dir='/Users/igyuseob/Desktop/capstone/data/052.구강 이미지 합성데이터/052.구강 이미지 합성데이터/3.개방데이터/1.데이터/Training/2.라벨링데이터',
    transform=get_transform(train=True)
)

val_dataset = ToothSegmentationDataset(
    image_dir='/Users/igyuseob/Desktop/capstone/data/052.구강 이미지 합성데이터/052.구강 이미지 합성데이터/3.개방데이터/1.데이터/Validation/1.원천데이터',
    label_dir='/Users/igyuseob/Desktop/capstone/data/052.구강 이미지 합성데이터/052.구강 이미지 합성데이터/3.개방데이터/1.데이터/Validation/2.라벨링데이터',
    transform=get_transform(train=False)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = get_model_instance_segmentation(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 학습 및 검증
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_one_epoch(model, optimizer, train_loader, DEVICE)
    evaluate(model, val_loader, DEVICE)

# 모델 저장
torch.save(model.state_dict(), "maskrcnn_resnet50_fpn_v2.pth")