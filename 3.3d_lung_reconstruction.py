import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button  # 用于交互按钮
import nibabel as nib
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 模型定义（保持不变）
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
        self.lung_classifier = nn.Conv2d(features[0], out_channels, 1)

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
        return self.lung_classifier(x)


def load_test_data(test_data_path):
    test_data = torch.load(test_data_path, weights_only=False)
    ct_volume = test_data["ct_volume"]
    lung_mask_volume = test_data["lung_mask_volume"]
    slice_count = test_data["slice_count"]
    original_height, original_width = ct_volume.shape[1], ct_volume.shape[2]
    print(f"Loaded test data: Original size (HxW)={original_height}x{original_width}, Slice count={slice_count}")
    return ct_volume, lung_mask_volume, slice_count, (original_height, original_width)


def preprocess_test_slices(ct_volume, img_size=224):
    preprocessed = []
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    for slice_img in ct_volume:
        augmented = transform(image=slice_img)
        preprocessed_slice = augmented["image"].numpy()
        preprocessed.append(preprocessed_slice)
    return np.stack(preprocessed), img_size


def postprocess_predictions(predictions, original_size):
    postprocessed = []
    original_height, original_width = original_size
    for pred in predictions:
        resized = cv2.resize(pred, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        postprocessed.append(resized)
    return np.stack(postprocessed)


def predict_volume(model, ct_volume, device, batch_size=8):
    model.eval()
    model.to(device)
    slice_count, channels, height, width = ct_volume.shape
    predictions = np.zeros((slice_count, height, width), dtype=np.float32)
    indices = list(range(slice_count))
    for i in tqdm(range(0, len(indices), batch_size), desc="Predicting slices"):
        batch_indices = indices[i:i + batch_size]
        batch = ct_volume[batch_indices]
        batch_tensor = torch.FloatTensor(batch).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                output = model(batch_tensor)
                output = torch.sigmoid(output)
                output = output.squeeze(1).cpu().numpy()
        predictions[batch_indices] = output
    return predictions


def reconstruct_3d(predictions, threshold=0.5):
    binary_mask = (predictions > threshold).astype(np.uint8)
    binary_mask = np.transpose(binary_mask, (1, 2, 0))
    print(f"Reconstructed 3D mask: Shape {binary_mask.shape}, Volume {np.sum(binary_mask)} voxels")
    print(f"Mask value range: {np.min(binary_mask)} ~ {np.max(binary_mask)}")
    if np.sum(binary_mask) > 0:
        print(f"Non-zero voxel positions: {np.argwhere(binary_mask > 0)[:5]}")
    else:
        print("Warning: Reconstructed mask is all zeros!")
    return binary_mask


def generate_point_cloud(mask_3d, voxel_spacing, downsample_ratio=0.01):
    z_indices, y_indices, x_indices = np.where(mask_3d > 0)
    x_coords = x_indices * voxel_spacing[0]
    y_coords = y_indices * voxel_spacing[1]
    z_coords = z_indices * voxel_spacing[2]
    point_cloud = np.column_stack([x_coords, y_coords, z_coords])
    print(f"Original point cloud: {len(point_cloud)} points")

    if downsample_ratio < 1.0 and len(point_cloud) > 0:
        sample_size = max(50, int(len(point_cloud) * downsample_ratio))
        indices = np.random.choice(len(point_cloud), size=sample_size, replace=False)
        downsampled_cloud = point_cloud[indices]
        print(f"Downsampled point cloud: {len(downsampled_cloud)} points (retained {downsample_ratio:.2%})")
        return downsampled_cloud
    return point_cloud


def visualize_3d_mask(point_cloud, title="3D Lung Point Cloud", output_dir=None):
    if point_cloud is None or len(point_cloud) == 0:
        print("No valid point cloud for visualization")
        return

    # 创建交互式3D图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    scatter = ax.scatter(
        point_cloud[:, 0],  # X
        point_cloud[:, 1],  # Y
        point_cloud[:, 2],  # Z
        c='green',
        alpha=0.6,
        s=5,
        marker='o'
    )

    # 计算并标记中心点
    center = np.mean(point_cloud, axis=0)
    ax.scatter(center[0], center[1], center[2], c='red', s=50, marker='*', label='Center')

    # 坐标轴标签
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)
    ax.legend()

    # 初始视角
    ax.view_init(elev=30, azim=45)

    # 自动保存图像（无需手动点击）
    if output_dir:
        auto_save_path = os.path.join(output_dir, "auto_saved_view.png")
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"自动保存视图至: {auto_save_path}")

    # 添加交互按钮（保留手动保存功能）
    ax_save = plt.axes([0.8, 0.01, 0.15, 0.05])
    btn_save = Button(ax_save, 'Save View')

    def save_view(event):
        if output_dir:
            img_path = os.path.join(output_dir, "interactive_view.png")
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            print(f"手动保存视图至: {img_path}")
    btn_save.on_clicked(save_view)

    # 显示交互式窗口
    plt.show()
    plt.close()


def calculate_dice_score(prediction, ground_truth):
    if prediction.shape != ground_truth.shape:
        ground_truth = np.transpose(ground_truth, (1, 2, 0))
    intersection = (prediction * ground_truth).sum()
    return (2.0 * intersection) / (prediction.sum() + ground_truth.sum() + 1e-8)


def save_prediction_as_nii(prediction, original_nii_path, output_path):
    original_nii = nib.load(original_nii_path)
    nii_img = nib.Nifti1Image(prediction, original_nii.affine, original_nii.header)
    nib.save(nii_img, output_path)
    print(f"Prediction saved to: {output_path}")


def calculate_lung_center(mask):
    coords = np.array(np.where(mask > 0)).T
    return np.mean(coords, axis=0).astype(int) if len(coords) > 0 else np.array([0, 0, 0])


def main():
    TEST_DATA_PATH = "./processed_data/test_data.pt"
    MODEL_PATH = "./saved_models/lung_model.pt"
    ORIGINAL_CT_PATH = "./9 CT scans/rp_im/9.nii.gz"
    OUTPUT_DIR = "./prediction_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    print(f"Loaded model from: {MODEL_PATH}")

    ct_volume, lung_mask_volume, slice_count, original_size = load_test_data(TEST_DATA_PATH)
    preprocessed_ct, img_size = preprocess_test_slices(ct_volume)
    print(f"Test data preprocessed: Resized to {img_size}x{img_size}")

    predictions = predict_volume(model, preprocessed_ct, device, batch_size=8)
    predictions = postprocess_predictions(predictions, original_size)
    print(f"Predictions post-processed: Restored to original size {original_size[0]}x{original_size[1]}")

    pred_3d_mask = reconstruct_3d(predictions)

    if lung_mask_volume is not None:
        original_dice = calculate_dice_score(pred_3d_mask, lung_mask_volume)
        print(f"Original resolution Dice score: {original_dice:.4f}")

    lung_center = calculate_lung_center(pred_3d_mask)
    print(f"Lung center voxel coordinates: {lung_center} (z, y, x)")

    voxel_spacing = nib.load(ORIGINAL_CT_PATH).header.get_zooms()
    print(f"Original voxel spacing (x, y, z): {voxel_spacing}")
    physical_center = (
        lung_center[2] * voxel_spacing[0],
        lung_center[1] * voxel_spacing[1],
        lung_center[0] * voxel_spacing[2]
    )
    print(f"Lung center physical coordinates (mm): {np.round(physical_center, 2)}")

    # 生成点云（1%采样率）
    point_cloud = generate_point_cloud(
        pred_3d_mask,
        voxel_spacing,
        downsample_ratio=0.01
    )

    # 交互式可视化（自动保存图像）
    visualize_3d_mask(
        point_cloud,
        title="3D Lung Segmentation (Interactive View)",
        output_dir=OUTPUT_DIR
    )

    # 保存结果
    if point_cloud is not None:
        np.save(os.path.join(OUTPUT_DIR, "lung_point_cloud.npy"), point_cloud)
        print(f"Point cloud data saved to: {os.path.join(OUTPUT_DIR, 'lung_point_cloud.npy')}")

    save_prediction_as_nii(
        pred_3d_mask,
        ORIGINAL_CT_PATH,
        os.path.join(OUTPUT_DIR, "predicted_lung_mask.nii.gz")
    )

    slices_dir = os.path.join(OUTPUT_DIR, "predicted_slices")
    os.makedirs(slices_dir, exist_ok=True)
    for i in range(slice_count):
        slice_path = os.path.join(slices_dir, f"slice_{i:03d}.png")
        plt.imsave(slice_path, predictions[i], cmap="gray")

    np.savetxt(os.path.join(OUTPUT_DIR, "lung_center.txt"), lung_center, fmt="%d")
    np.savetxt(os.path.join(OUTPUT_DIR, "lung_center_physical.txt"), physical_center, fmt="%.2f")
    print("Prediction and reconstruction completed!")


if __name__ == "__main__":
    main()