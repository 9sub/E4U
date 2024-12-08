import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm

# Dataset 정의
class ToothSegmentationDataset(Dataset):
    def __init__(self, data_dir, is_train='', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.label_paths = []
        self.load_data_paths()

    def load_data_paths(self):
        for view_dir in ['1.right', '2.front', '3.left', '4.upper', '5.lower']:
            view_path = os.path.join(self.data_dir, view_dir)
            json_files = [f for f in os.listdir(view_path) if f.endswith('.json')]

            for json_file in tqdm(json_files, desc=f"Processing {view_dir}", leave=False):
                with open(os.path.join(view_path, json_file)) as f:
                    annotation = json.load(f)
                    base_name = json_file.replace('.json', '')
                    image_path = f"/Users/igyuseob/Desktop/capstone/data/052.구강 이미지 합성데이터/3.개방데이터/1.데이터/{self.is_train}/1.원천데이터/{view_dir}/{base_name}.png"
                    if os.path.exists(image_path):
                        self.image_paths.append(image_path)
                        self.label_paths.append(annotation)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.parse_json_to_mask(self.label_paths[idx], image.size)
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def parse_json_to_mask(self, annotation, img_size):
        mask = np.zeros(img_size, dtype=np.uint8)
        for tooth in annotation['tooth']:
            points = tooth['segmentation']
            cv2.fillPoly(mask, [np.array(points, np.int32)], 1)
        return Image.fromarray(mask)

# 데이터셋 및 DataLoader 생성
data_dir = "/path/to/data"
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = ToothSegmentationDataset(data_dir, is_train="Training", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Pre-trained U-Net 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

print("Training complete.")