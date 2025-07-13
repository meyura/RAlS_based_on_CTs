import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 允许NumPy相关全局对象
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype
])


# ----------------------
# 数据集类（仅保留肺部相关数据）
# ----------------------
class CTDataset(Dataset):
    def __init__(self, data_path, augment=False):
        self.data = torch.load(data_path, weights_only=False)
        self.ct_slices = self.data[0]
        self.lung_masks = self.data[1]  # 仅保留肺部掩码
        self.augment = augment

        print(f"总样本数: {len(self.ct_slices)}（仅用于肺部分割）")

    def __len__(self):
        return len(self.ct_slices)

    def __getitem__(self, idx):
        # 加载样本（仅加载CT和肺部掩码）
        ct = self.ct_slices[idx]
        lung_mask = self.lung_masks[idx]

        # 数据增强
        if self.transform is not None:
            augmented = self.transform(image=ct, mask=lung_mask)
            ct = augmented["image"]
            lung_mask = augmented["mask"]

        # 转换为张量
        lung_mask = torch.as_tensor(lung_mask, dtype=torch.float32).clone().detach()

        return ct, lung_mask  # 只返回CT和肺部掩码

    @property
    def transform(self):
        img_size = 224
        if self.augment:
            return A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Rotate(limit=35, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                ToTensorV2(),
            ])


# ----------------------
# 模型定义（仅保留肺部分支）
# ----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        features = [128, 256, 512, 1024]
        self.encoders = nn.ModuleList([
            DoubleConv(in_channels, features[0]),
            DoubleConv(features[0], features[1]),
            DoubleConv(features[1], features[2]),
            DoubleConv(features[2], features[3]),
        ])
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(features[3] * 2, features[3], 2, 2),
            nn.ConvTranspose2d(features[3], features[2], 2, 2),
            nn.ConvTranspose2d(features[2], features[1], 2, 2),
            nn.ConvTranspose2d(features[1], features[0], 2, 2),
        ])
        self.decoders = nn.ModuleList([
            DoubleConv(features[3] * 2, features[3]),
            DoubleConv(features[3], features[2]),
            DoubleConv(features[2], features[1]),
            DoubleConv(features[1], features[0]),
        ])
        self.lung_classifier = nn.Conv2d(features[0], out_channels, 1)  # 仅保留肺部分支

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            skip = skips[idx]
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])
            x = decoder(torch.cat((skip, x), dim=1))
        return self.lung_classifier(x)  # 仅输出肺部预测


# ----------------------
# 损失函数和指标（仅针对肺部）
# ----------------------
def dice_score(pred, target, smooth=1e-5):
    pred = pred.contiguous().float()
    target = target.contiguous().float()
    intersection = (pred * target).sum(dim=(1, 2))
    pred_sum = pred.sum(dim=(1, 2))
    target_sum = target.sum(dim=(1, 2))
    return ((2. * intersection + smooth) / (pred_sum + target_sum + smooth)).mean()


def bce_dice_loss(pred, target, weight_positive=8.0):
    """针对肺部的损失函数（无需过高权重，因肺部占比高）"""
    bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    weighted_bce = (target * weight_positive * bce + (1 - target) * bce).mean()
    pred_sigmoid = torch.sigmoid(pred)
    dice = dice_score(pred_sigmoid, target)
    return weighted_bce + (1 - dice)


# ----------------------
# 训练和验证函数（仅肺部）
# ----------------------
def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0.0
    total_lung_dice = 0.0
    loop = tqdm(loader, desc="Training")

    for ct, lung_mask in loop:
        ct, lung_mask = ct.to(device), lung_mask.to(device)
        with torch.amp.autocast(device_type='cuda'):
            lung_pred = model(ct)  # 仅输出肺部预测
            loss = loss_fn(lung_pred.squeeze(1), lung_mask)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 计算指标
        lung_pred_sigmoid = torch.sigmoid(lung_pred.squeeze(1))
        lung_dice = dice_score(lung_pred_sigmoid, lung_mask)
        total_loss += loss.item()
        total_lung_dice += lung_dice.item()
        loop.set_postfix(loss=total_loss/(loop.n+1), lung_dice=total_lung_dice/(loop.n+1))

    return total_loss/len(loader), total_lung_dice/len(loader)


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_lung_dice = 0.0
    with torch.no_grad():
        for ct, lung_mask in loader:
            ct, lung_mask = ct.to(device), lung_mask.to(device)
            lung_pred = model(ct)
            loss = loss_fn(lung_pred.squeeze(1), lung_mask)
            lung_dice = dice_score(torch.sigmoid(lung_pred.squeeze(1)), lung_mask)
            total_loss += loss.item()
            total_lung_dice += lung_dice.item()
    return total_loss/len(loader), total_lung_dice/len(loader)


# ----------------------
# 主函数（仅训练肺部）
# ----------------------
def main():
    # 配置路径
    TRAIN_DATA_PATH = "./processed_data/Training_data.pt"
    VAL_DATA_PATH = "./processed_data/Validation_data.pt"
    SAVE_DIR = "./saved_models"
    os.makedirs(SAVE_DIR, exist_ok=True)
    LUNG_MODEL_PATH = os.path.join(SAVE_DIR, "lung_model.pt")  # 仅保存肺部模型
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据集（仅肺部）
    train_dataset = CTDataset(TRAIN_DATA_PATH, augment=True)
    val_dataset = CTDataset(VAL_DATA_PATH, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 初始化模型、优化器
    model = UNet(in_channels=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler()
    best_lung_dice = 0.0
    EPOCHS = 80  # 专注肺部训练，80轮足够收敛

    # 训练曲线记录
    train_metrics = {"loss": [], "lung_dice": []}
    val_metrics = {"loss": [], "lung_dice": []}

    # 训练循环
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)

        # 动态调整学习率
        if epoch > 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        if epoch > 60:
            param_group['lr'] = 1e-6

        # 训练和验证
        train_loss, train_lung_dice = train_one_epoch(
            model, train_loader, optimizer, bce_dice_loss, scaler, DEVICE
        )
        val_loss, val_lung_dice = validate(
            model, val_loader, bce_dice_loss, DEVICE
        )

        # 记录指标
        train_metrics["loss"].append(train_loss)
        train_metrics["lung_dice"].append(train_lung_dice)
        val_metrics["loss"].append(val_loss)
        val_metrics["lung_dice"].append(val_lung_dice)

        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Lung Dice: {train_lung_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Lung Dice: {val_lung_dice:.4f}")

        # 保存最佳模型
        if val_lung_dice > best_lung_dice:
            best_lung_dice = val_lung_dice
            torch.save(model.state_dict(), LUNG_MODEL_PATH)
            print(f"最佳肺部模型已保存至 {LUNG_MODEL_PATH}（Dice: {best_lung_dice:.4f}）")

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics["loss"], label="Train")
    plt.plot(val_metrics["loss"], label="Val")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 肺部Dice曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics["lung_dice"], label="Train")
    plt.plot(val_metrics["lung_dice"], label="Val")
    plt.title("Lung Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    plt.tight_layout()
    plt.savefig("lung_training_curves.png")
    print("肺部训练曲线已保存至 lung_training_curves.png")
    print("肺部分割模型训练完成！")


if __name__ == "__main__":
    main()