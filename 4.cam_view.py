import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
import cv2


# ------------------- 1. 数据加载函数 -------------------
def load_center_point(file_path):
    if os.path.exists(file_path):
        center = np.loadtxt(file_path)
        print(f"Loaded lung center from {file_path}: {center}")
        return center
    else:
        print(f"Error: {file_path} does not exist!")
        return np.array([0, 0, 0])


def load_3d_mask(mask_path):
    if mask_path.endswith('.nii.gz'):
        nii_img = nib.load(mask_path)
        mask = nii_img.get_fdata()
        spacing = nii_img.header.get_zooms()
        print(f"Loaded 3D mask: shape {mask.shape}, voxel spacing {spacing}")
        return mask, spacing
    else:
        print(f"Error: Unsupported mask format")
        return None, None


# ------------------- 2. 点云生成与透视投影函数 -------------------
def generate_point_cloud_and_projection(mask, spacing, downsample_ratio=0.01, fov=1000):
    z_indices, y_indices, x_indices = np.where(mask > 0)
    x_coords = x_indices * spacing[0]
    y_coords = y_indices * spacing[1]
    z_coords = z_indices * spacing[2]
    point_cloud = np.column_stack([x_coords, y_coords, z_coords])

    print(f"Original point count: {len(point_cloud)}")
    sample_size = max(100, int(len(point_cloud) * downsample_ratio))
    indices = np.random.choice(len(point_cloud), size=sample_size, replace=False)
    downsampled = point_cloud[indices]
    print(f"Downsampled point count: {len(downsampled)}")

    temp_pixel = np.zeros((len(downsampled), 2))
    for i in range(len(downsampled)):
        x, y, z = downsampled[i]
        z = max(z, 1e-6)
        temp_pixel[i, 0] = (x * fov) / z
        temp_pixel[i, 1] = (y * fov) / z

    cx, cy = np.mean(temp_pixel[:, 0]), np.mean(temp_pixel[:, 1])

    pixel_coords = np.zeros((len(downsampled), 2))
    for i in range(len(downsampled)):
        x, y, z = downsampled[i]
        z = max(z, 1e-6)
        pixel_coords[i, 0] = (x * fov) / z + cx
        pixel_coords[i, 1] = (y * fov) / z + cy

    return downsampled, pixel_coords


# ------------------- 3. 空间分布选点函数（增强版） -------------------
def select_spatially_distributed_points_improved(point_cloud, pixel_coords, n=8, min_dist_2d=40, min_dist_3d=50):
    indices = np.arange(len(point_cloud))
    selected_indices = []

    first_idx = np.random.choice(indices)
    selected_indices.append(first_idx)

    for _ in range(n - 1):
        distances_2d = np.zeros(len(indices))
        distances_3d = np.zeros(len(indices))

        for i, idx in enumerate(indices):
            if idx in selected_indices:
                distances_2d[i] = -1
                distances_3d[i] = -1
                continue

            dist_2d = np.min([
                np.sqrt(np.sum((pixel_coords[idx] - pixel_coords[s_idx]) ** 2))
                for s_idx in selected_indices
            ])

            dist_3d = np.min([
                np.sqrt(np.sum((point_cloud[idx] - point_cloud[s_idx]) ** 2))
                for s_idx in selected_indices
            ])

            distances_2d[i] = dist_2d
            distances_3d[i] = dist_3d

        valid_mask = (distances_2d >= min_dist_2d) & (distances_3d >= min_dist_3d)
        valid_indices = indices[valid_mask]

        if len(valid_indices) > 0:
            next_idx = np.random.choice(valid_indices)
            selected_indices.append(next_idx)
        else:
            next_idx = indices[np.argmax(distances_2d)]
            selected_indices.append(next_idx)
            print(f"Warning: 无法找到满足条件的点，选择2D距离最远的点 {next_idx}")

    return point_cloud[selected_indices], pixel_coords[selected_indices]


# ------------------- 4. 变换矩阵与投影函数 -------------------
def compute_transformation_matrix(selected_3d, selected_2d, fov):
    cx, cy = np.mean(selected_2d[:, 0]), np.mean(selected_2d[:, 1])
    fx = fy = fov
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([0.01, -0.01, 0, 0, 0], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        selected_3d, selected_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
    )
    if not success:
        raise ValueError("PNP求解失败，无法计算变换矩阵")

    R, _ = cv2.Rodrigues(rvec)
    return camera_matrix, R, tvec, dist_coeffs


def project_3d_to_2d(point_3d, camera_matrix, R, tvec, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    point_2d, _ = cv2.projectPoints(
        np.array([point_3d], dtype=np.float32),
        R, tvec, camera_matrix, dist_coeffs
    )
    return point_2d[0][0]


def back_project_2d_to_3d(point_2d, camera_matrix, R, tvec, depth):
    u, v = point_2d
    fx, _, cx = camera_matrix[0]
    _, fy, cy = camera_matrix[1]
    Xc = (u - cx) * depth / fx
    Yc = (v - cy) * depth / fy
    Zc = depth
    point_cam = np.array([Xc, Yc, Zc])

    R_inv = R.T
    point_world = R_inv @ (point_cam - tvec.flatten())
    return point_world


# ------------------- 5. 可视化函数（修复变量作用域） -------------------
def visualize_with_points(point_cloud, pixel_coords, selected_pc, selected_px,
                          center_3d, center_2d, predicted_3d, camera_matrix, R, tvec, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 2D视图
    plt.figure(figsize=(10, 8))
    plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c='blue', alpha=0.3, s=2, label='Point cloud')

    for i in range(len(selected_px)):
        plt.scatter(selected_px[i, 0], selected_px[i, 1],
                    c=f'C{i}', s=60, edgecolors='black', label=f'Point {i + 1}')

    plt.scatter(center_2d[0], center_2d[1], c='red', s=120,
                edgecolors='yellow', marker='o', label='Original Center (2D)')

    predicted_2d = project_3d_to_2d(predicted_3d, camera_matrix, R, tvec)
    plt.scatter(predicted_2d[0], predicted_2d[1], c='purple', s=120,
                marker='x', edgecolors='black', label='Back-projected Center (2D)')

    error_2d = np.sqrt(np.sum((center_2d - predicted_2d) ** 2))
    plt.arrow(center_2d[0], center_2d[1],
              predicted_2d[0] - center_2d[0],
              predicted_2d[1] - center_2d[1],
              color='orange', head_width=5, head_length=5,
              length_includes_head=True, label=f'2D Error: {error_2d:.2f} px')

    plt.xlabel('X (px)')
    plt.ylabel('Y (px)')
    plt.title('2D View (Perspective Projection)')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_view.png'), dpi=300)
    plt.close()

    # 2. 3D视图
    fig = plt.figure(figsize=(12, 10))
    ax3d = fig.add_subplot(111, projection='3d')

    ax3d.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                 c='gray', alpha=0.3, s=5, label='Point cloud')

    for i in range(len(selected_pc)):
        ax3d.scatter(selected_pc[i, 0], selected_pc[i, 1], selected_pc[i, 2],
                     c=f'C{i}', s=80, edgecolors='black', label=f'Point {i + 1}')

    ax3d.scatter(center_3d[0], center_3d[1], center_3d[2],
                 c='green', s=150, marker='o', edgecolors='black', label='Original Center (3D)')
    ax3d.scatter(predicted_3d[0], predicted_3d[1], predicted_3d[2],
                 c='red', s=150, marker='*', edgecolors='black', label='Back-projected Center (3D)')

    error_3d = np.linalg.norm(predicted_3d - center_3d)
    ax3d.quiver(center_3d[0], center_3d[1], center_3d[2],
                predicted_3d[0] - center_3d[0],
                predicted_3d[1] - center_3d[1],
                predicted_3d[2] - center_3d[2],
                color='orange', arrow_length_ratio=0.1,
                label=f'3D Error: {error_3d:.2f} mm')

    ax3d.set_xlabel('X (mm)')
    ax3d.set_ylabel('Y (mm)')
    ax3d.set_zlabel('Z (mm)')
    ax3d.set_title('3D View (Original vs Back-projected)')
    ax3d.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_view.png'), dpi=300)
    plt.close()

    print(f"可视化结果已保存至: {output_dir}")


# ------------------- 6. 主函数（完整流程） -------------------
def main():
    mask_path = "./prediction_results/predicted_lung_mask.nii.gz"
    center_path = "./prediction_results/lung_center_physical.txt"
    output_dir = "./point_cloud_results"
    fov = 1000
    select_min_dist_2d = 40
    select_min_dist_3d = 50
    num_selected_points = 8

    center_3d = load_center_point(center_path)
    mask, spacing = load_3d_mask(mask_path)

    if mask is not None:
        point_cloud, pixel_coords = generate_point_cloud_and_projection(
            mask, spacing, downsample_ratio=0.01, fov=fov
        )

        selected_pc, selected_px = select_spatially_distributed_points_improved(
            point_cloud, pixel_coords, n=num_selected_points,
            min_dist_2d=select_min_dist_2d, min_dist_3d=select_min_dist_3d
        )

        camera_matrix, R, tvec, dist_coeffs = compute_transformation_matrix(selected_pc, selected_px, fov=fov)
        print("\n相机内参 K:\n", camera_matrix.round(2))
        print("旋转矩阵 R:\n", R.round(4))
        print("平移向量 t:\n", tvec.flatten().round(2))
        print("畸变系数:\n", dist_coeffs.flatten().round(4))

        center_2d = project_3d_to_2d(center_3d, camera_matrix, R, tvec, dist_coeffs)
        print(f"\n中心点3D坐标: {center_3d.round(2)} mm")
        print(f"中心点2D投影坐标: {center_2d.round(2)} px")

        depth = center_3d[2]
        predicted_center_3d = back_project_2d_to_3d(center_2d, camera_matrix, R, tvec, depth=depth)
        print(f"反投影的中心点3D坐标: {predicted_center_3d.round(2)} mm")

        error_3d = np.linalg.norm(predicted_center_3d - center_3d)
        print(f"3D反投影误差: {error_3d:.2f} mm")

        # 修复：将camera_matrix, R, tvec传入可视化函数
        visualize_with_points(point_cloud, pixel_coords, selected_pc, selected_px,
                              center_3d, center_2d, predicted_center_3d,
                              camera_matrix, R, tvec, output_dir)

        np.savez(os.path.join(output_dir, '8_selected_points.npz'),
                 pixel=selected_px, world=selected_pc,
                 center_3d=center_3d, center_2d=center_2d,
                 predicted_3d=predicted_center_3d,
                 camera_matrix=camera_matrix, R=R, tvec=tvec,
                 dist_coeffs=dist_coeffs)

    else:
        print("加载mask失败，程序退出")


if __name__ == "__main__":
    main()