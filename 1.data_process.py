import os
import numpy as np
import cv2
import torch
import nibabel as nib
from sklearn.model_selection import train_test_split

# 设置参数
INPUT_DIR = "./9 CT scans/"
OUTPUT_DIR = "./processed_data/"
TEST_SIZE = 0.3  # 验证集占比
RANDOM_SEED = 42

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_volume(vol_name):
    """加载并预处理单个CT容积"""
    ct_path = os.path.join(INPUT_DIR, "rp_im", vol_name)
    lung_mask_path = os.path.join(INPUT_DIR, "rp_lung_msk", vol_name)

    # 加载数据
    ct_vol = nib.load(ct_path).get_fdata()  # 初始为float64
    lung_mask_vol = nib.load(lung_mask_path).get_fdata()

    # 统一旋转
    for z in range(ct_vol.shape[2]):
        ct_vol[:, :, z] = cv2.rotate(ct_vol[:, :, z], cv2.ROTATE_90_CLOCKWISE)
        lung_mask_vol[:, :, z] = cv2.rotate(lung_mask_vol[:, :, z], cv2.ROTATE_90_CLOCKWISE)

    # 归一化到0-255并强制转换为8位无符号整数
    for z in range(ct_vol.shape[2]):
        ct_slice = ct_vol[:, :, z].astype(np.float32)
        ct_slice = cv2.normalize(
            ct_slice, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        ct_vol[:, :, z] = ct_slice

    # 掩码二值化并转为uint8
    lung_mask_vol = np.where(lung_mask_vol > 0, 1, 0).astype(np.uint8)

    return ct_vol, lung_mask_vol


def split_slices_into_2d(ct_vol, lung_mask_vol):
    """拆分3D容积为2D切片，并转为RGB"""
    slices_ct = []
    slices_lung = []
    for z in range(ct_vol.shape[2]):
        ct_slice = ct_vol[:, :, z].astype(np.uint8)
        ct_slice_rgb = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2RGB)
        slices_ct.append(ct_slice_rgb)
        slices_lung.append(lung_mask_vol[:, :, z])
    return slices_ct, slices_lung


def main():
    # 获取所有CT容积并排序
    ct_volumes = sorted(
        os.listdir(os.path.join(INPUT_DIR, "rp_im")),
        key=lambda x: int(x.split(".")[0])
    )
    print(f"发现 {len(ct_volumes)} 个CT容积")

    # 最后一个容积作为测试集
    test_vol_name = ct_volumes[-1]
    train_val_vols = ct_volumes[:-1]
    print(f"将最后一个容积 {test_vol_name} 作为测试集")
    print(f"剩余 {len(train_val_vols)} 个容积用于划分训练集和验证集")

    # 收集训练和验证集的所有切片
    all_ct = []
    all_lung = []
    for vol_name in train_val_vols:
        print(f"预处理训练/验证集容积：{vol_name}")
        ct_vol, lung_mask_vol = load_and_preprocess_volume(vol_name)
        slices_ct, slices_lung = split_slices_into_2d(ct_vol, lung_mask_vol)
        all_ct.extend(slices_ct)
        all_lung.extend(slices_lung)

    # 划分训练集和验证集
    print(f"训练/验证集总切片数：{len(all_ct)}，按 {1-TEST_SIZE}:{TEST_SIZE} 划分")
    train_ct, val_ct, train_lung, val_lung = train_test_split(
        all_ct, all_lung,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    # 保存训练集
    train_data = (train_ct, train_lung)
    train_path = os.path.join(OUTPUT_DIR, "Training_data.pt")
    torch.save(train_data, train_path)
    print(f"训练集保存至 {train_path}，包含 {len(train_ct)} 个切片")

    # 保存验证集
    val_data = (val_ct, val_lung)
    val_path = os.path.join(OUTPUT_DIR, "Validation_data.pt")
    torch.save(val_data, val_path)
    print(f"验证集保存至 {val_path}，包含 {len(val_ct)} 个切片")

    # 保存测试集（3D结构）
    print(f"处理测试集容积：{test_vol_name}")
    ct_vol, lung_mask_vol = load_and_preprocess_volume(test_vol_name)
    ct_vol_rgb = np.stack([cv2.cvtColor(ct_vol[:, :, z].astype(np.uint8), cv2.COLOR_GRAY2RGB)
                          for z in range(ct_vol.shape[2])])
    test_data = {
        "ct_volume": ct_vol_rgb,
        "lung_mask_volume": lung_mask_vol,
        "slice_count": ct_vol.shape[2]
    }
    test_path = os.path.join(OUTPUT_DIR, "test_data.pt")
    torch.save(test_data, test_path)
    print(f"测试集保存至 {test_path}，包含 {ct_vol.shape[2]} 个切片")

    print("数据处理完成")


if __name__ == "__main__":
    main()