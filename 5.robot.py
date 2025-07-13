import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from numpy.linalg import norm, pinv
import cv2
from sklearn.decomposition import PCA


class SimpleRobotArm:
    def __init__(self, base_position=np.array([0, 0, 800], dtype=np.float64)):
        self.base_position = base_position.astype(np.float64)
        self.link_lengths = np.array([250, 300, 200, 150, 100, 100], dtype=np.float64)
        self.max_reach = np.sum(self.link_lengths) * 0.95  # ~1045mm
        self.joint_angles = np.array([0.0, -0.2, 0.3, -0.1, 0.0, 0.0], dtype=np.float64)
        self.target_position = None
        self.animation_frames = []
        self.point_cloud_world = None
        self.update_pose()
        print(f"初始末端位置: {self.end_effector_position.round(2)} mm")
        print(f"机械臂最大工作半径: {self.max_reach:.2f} mm")
        print(f"基座位置: {self.base_position.round(2)} mm")

    def update_pose(self):
        self.joint_positions = [self.base_position.copy()]
        current_pos = self.base_position.copy()
        current_rotation = np.eye(3)

        for i in range(6):
            theta = self.joint_angles[i]
            length = self.link_lengths[i]

            if i % 2 == 0:  # 绕Z轴旋转
                R = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
            else:  # 绕Y轴旋转
                R = np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])

            current_rotation = current_rotation @ R
            direction = current_rotation @ np.array([length, 0, 0])
            current_pos += direction
            self.joint_positions.append(current_pos.copy())

        self.end_effector_position = self.joint_positions[-1]
        if self.target_position is not None:
            self.animation_frames.append((np.array(self.joint_positions), self.end_effector_position.copy()))

    def forward_kinematics(self, angles):
        current_pos = self.base_position.copy()
        current_rotation = np.eye(3)

        for i in range(6):
            theta = angles[i]
            length = self.link_lengths[i]

            if i % 2 == 0:
                R = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
            else:
                R = np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])

            current_rotation = current_rotation @ R
            direction = current_rotation @ np.array([length, 0, 0])
            current_pos += direction

        return current_pos

    def inverse_kinematics(self, target_position, max_iterations=1000, tolerance=1e-5):
        target = np.array(target_position, dtype=np.float64)
        target_rel = target - self.base_position
        dist_from_base = norm(target_rel)

        if dist_from_base > self.max_reach:
            print(f"目标点超出工作空间（距离基座 {dist_from_base:.2f} mm），调整中...")
            dir_norm = target_rel / dist_from_base
            z_weight = 1.2
            dir_scaled = dir_norm * np.array([1.0, 1.0, z_weight])
            dir_scaled = dir_scaled / norm(dir_scaled)
            target_proj = self.base_position + dir_scaled * self.max_reach
            z_error = target[2] - self.base_position[2]
            target_proj[2] = self.base_position[2] + min(z_error, self.max_reach * 0.8)
            target = target_proj
            print(f"调整后目标点: {target.round(2)} mm")

        initial_guess = self.joint_angles.copy()

        def objective(angles):
            end_pos = self.forward_kinematics(angles)
            pos_error = norm((end_pos - target) * np.array([1.0, 1.0, 1.5]))
            movement_cost = norm(angles - initial_guess) * 0.3
            joint_diff_cost = norm(np.diff(angles)) * 0.2
            self.joint_angles = angles
            jacobian = self.compute_jacobian()
            singularity_cost = 1.0 / (1e-6 + np.linalg.det(jacobian @ jacobian.T)) * 0.01
            return pos_error + movement_cost + joint_diff_cost + singularity_cost

        bounds = [
            (-np.pi * 0.7, np.pi * 0.7),
            (-np.pi * 0.6, np.pi * 0.3),
            (-np.pi * 0.5, np.pi * 0.5),
            (-np.pi * 0.6, np.pi * 0.6),
            (-np.pi * 0.5, np.pi * 0.5),
            (-np.pi, np.pi)
        ]

        result = minimize(
            objective, initial_guess, method='BFGS',
            options={'maxiter': max_iterations, 'gtol': tolerance, 'eps': 1e-8}
        )

        if result.success:
            print(f"优化成功，迭代次数: {result.nit}，最终误差: {result.fun:.6f}")
            return result.x, target
        else:
            print(f"优化未成功: {result.message}，误差: {result.fun:.6f}")
            return result.x, target

    def compute_jacobian(self):
        end_pos = self.end_effector_position
        jacobian = np.zeros((3, 6))

        for i in range(6):
            joint_pos = self.joint_positions[i]
            r = end_pos - joint_pos
            if i % 2 == 0:
                axis = np.array([0, 0, 1])
            else:
                axis = np.array([0, 1, 0])
            jacobian[:, i] = np.cross(axis, r)

        return jacobian

    def move_to_target(self, target_position, steps=150):
        self.target_position = np.array(target_position, dtype=np.float64)
        self.animation_frames = []
        initial_angles = self.joint_angles.copy()  # 正确的变量名
        target_angles, adjusted_target = self.inverse_kinematics(self.target_position)
        self.target_position = adjusted_target

        # 三次样条插值生成平滑轨迹
        key_times = np.linspace(0, 1, 5)
        key_angles = np.zeros((5, 6))
        for i in range(6):
            key_angles[:, i] = np.linspace(initial_angles[i], target_angles[i], 5)

        t_interp = np.linspace(0, 1, steps)
        angles_interp = np.zeros((steps, 6))
        for i in range(6):
            spline = interp1d(key_times, key_angles[:, i], kind='cubic')
            angles_interp[:, i] = spline(t_interp)

        # 按插值后的角度运动
        for t_idx in range(steps):
            self.joint_angles = angles_interp[t_idx]
            self.update_pose()

            if t_idx % 20 == 0:
                dist = norm(self.end_effector_position - self.target_position)
                # 修正变量名：initial_guess → initial_angles
                angle_change = norm(self.joint_angles - initial_angles)
                print(f"帧 {t_idx}: 距离目标 {dist:.2f} mm，关节总变化 {angle_change:.3f} rad")

    def animate(self, save_path="robot_trajectory.gif", low_res=True):
        if not self.animation_frames:
            print("无动画数据，请先调用move_to_target")
            return

        dpi = 100 if low_res else 150
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        link_lines = [ax.plot([], [], [], 'b-', linewidth=3)[0] for _ in range(6)]
        joint_scatter = ax.scatter([], [], [], c='r', s=60, edgecolors='k', label='Joints')
        end_scatter = ax.scatter([], [], [], c='g', s=120, marker='*', edgecolors='k', label='End Effector')
        target_scatter = ax.scatter(
            [self.target_position[0]], [self.target_position[1]], [self.target_position[2]],
            c='m', s=150, marker='o', edgecolors='k', label='Adjusted Target'
        )
        original_center_scatter = ax.scatter(
            [self.original_center_world[0]], [self.original_center_world[1]], [self.original_center_world[2]],
            c='blue', s=150, marker='x', label='Original Center'
        )

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = self.base_position[0] + self.max_reach * np.cos(u) * np.sin(v)
        y = self.base_position[1] + self.max_reach * np.sin(u) * np.sin(v)
        z = self.base_position[2] + self.max_reach * np.cos(v)
        ax.plot_wireframe(x, y, z, color='red', alpha=0.2, linewidth=1, label='Workspace')

        if self.point_cloud_world is not None:
            ax.scatter(
                self.point_cloud_world[:, 0], self.point_cloud_world[:, 1], self.point_cloud_world[:, 2],
                c='purple', s=20, alpha=0.5, marker='^', label='Points'
            )

        path_line = ax.plot([], [], [], 'g--', linewidth=2, alpha=0.6, label='Path')

        all_points = np.concatenate([frame[0] for frame in self.animation_frames] +
                                    [self.target_position[None, :], self.original_center_world[None, :]])
        max_range = np.max(np.ptp(all_points, axis=0)) * 1.2
        mid = np.mean(all_points, axis=0)
        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Robot Arm with Smooth Trajectory')
        ax.legend(loc='upper left', fontsize=8)

        def update(frame):
            joints, end = self.animation_frames[frame]
            for i in range(6):
                link_lines[i].set_data([joints[i, 0], joints[i + 1, 0]], [joints[i, 1], joints[i + 1, 1]])
                link_lines[i].set_3d_properties([joints[i, 2], joints[i + 1, 2]])
            joint_scatter._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
            end_scatter._offsets3d = (end[0:1], end[1:2], end[2:3])
            path_x = [f[1][0] for f in self.animation_frames[:frame + 1]]
            path_y = [f[1][1] for f in self.animation_frames[:frame + 1]]
            path_z = [f[1][2] for f in self.animation_frames[:frame + 1]]
            path_line[0].set_data(path_x, path_y)
            path_line[0].set_3d_properties(path_z)
            return link_lines + [joint_scatter, end_scatter, target_scatter, path_line[0]]

        ani = FuncAnimation(fig, update, frames=len(self.animation_frames),
                            interval=80, blit=True, repeat=True)

        try:
            ani.save(save_path, writer='pillow', fps=15, dpi=dpi)
            print(f"动画已保存至: {save_path}")
        except Exception as e:
            print(f"保存动画失败: {e}")
            plt.show()
        finally:
            plt.close()


# ------------------- 其他函数保持不变 -------------------
def analyze_point_cloud_quality(world_coords):
    min_dist = float('inf')
    for i in range(len(world_coords)):
        for j in range(i + 1, len(world_coords)):
            dist = norm(world_coords[i] - world_coords[j])
            if dist < min_dist:
                min_dist = dist

    ranges = np.ptp(world_coords, axis=0)
    spread_ratio = max(ranges) / min(ranges) if min(ranges) > 0 else float('inf')

    pca = PCA(n_components=3)
    pca.fit(world_coords)
    variance_ratio = pca.explained_variance_ratio_
    planar_score = variance_ratio[2] / variance_ratio[0]

    print("\n点云质量分析:")
    print(f"点间最小距离: {min_dist:.2f} mm")
    print(f"各维度范围: X={ranges[0]:.2f} mm, Y={ranges[1]:.2f} mm, Z={ranges[2]:.2f} mm")
    print(f"分布范围比率: {spread_ratio:.2f}")
    print(f"PCA方差比率: {variance_ratio.round(4)}")
    print(f"平面度分数: {planar_score:.6f} (越小越平面)")

    if planar_score < 0.01:
        print("警告: 点云分布接近平面，可能导致变换矩阵求解不稳定！")

    return {
        'min_distance': min_dist,
        'ranges': ranges,
        'spread_ratio': spread_ratio,
        'variance_ratio': variance_ratio,
        'planar_score': planar_score
    }


def pixel_to_world(pixel_coords, world_coords, center_pixel, original_depth,
                   camera_matrix=None, dist_coeffs=None, use_depth_constraint=True):
    quality = analyze_point_cloud_quality(world_coords)

    if camera_matrix is None:
        min_px, max_px = np.min(pixel_coords), np.max(pixel_coords)
        fx = fy = max_px - min_px
        cx, cy = np.mean(pixel_coords, axis=0)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        avg_depth = np.mean(world_coords[:, 2])
        pixel_range = np.ptp(pixel_coords, axis=0)
        world_range = np.ptp(world_coords[:, :2], axis=0)
        fx2 = avg_depth * pixel_range[0] / world_range[0]
        fy2 = avg_depth * pixel_range[1] / world_range[1]
        camera_matrix[0, 0] = (fx + fx2) / 2
        camera_matrix[1, 1] = (fy + fy2) / 2

        print("\n估算的相机内参:")
        print(f"焦距: fx={camera_matrix[0, 0]:.2f}, fy={camera_matrix[1, 1]:.2f}")
        print(f"图像中心: cx={camera_matrix[0, 2]:.2f}, cy={camera_matrix[1, 2]:.2f}")

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1))

    success, rvec, tvec = cv2.solvePnP(
        world_coords, pixel_coords, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
    )
    if not success:
        print("警告: SQPNP求解失败，尝试使用EPNP...")
        success, rvec, tvec = cv2.solvePnP(
            world_coords, pixel_coords, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            raise ValueError("PNP求解失败，无法计算变换矩阵")

    R, _ = cv2.Rodrigues(rvec)
    print("\n求解的变换矩阵（旋转矩阵）:\n", R.round(4))
    print("求解的平移向量:\n", tvec.flatten().round(4))

    u, v = center_pixel
    fx, _, cx = camera_matrix[0]
    _, fy, cy = camera_matrix[1]

    X_cam = (u - cx) / fx * original_depth
    Y_cam = (v - cy) / fy * original_depth
    Z_cam = original_depth
    center_cam = np.array([X_cam, Y_cam, Z_cam])
    center_world = np.dot(R.T, center_cam - tvec.flatten())

    if quality['planar_score'] < 0.01 and use_depth_constraint:
        print("检测到平面点云，应用坐标系对齐优化...")
        cloud_center = np.mean(world_coords, axis=0)
        R_aligned = R.copy()
        tvec_aligned = tvec.copy()

        z_align_factor = 0.5
        R_aligned[2, :] = R[2, :] * z_align_factor
        R_aligned = R_aligned / np.linalg.norm(R_aligned, axis=1)[:, np.newaxis]

        center_world_aligned = np.dot(R_aligned.T, center_cam - tvec_aligned.flatten())
        error_original = norm(center_world - cloud_center)
        error_aligned = norm(center_world_aligned - cloud_center)

        if error_aligned < error_original:
            print(f"应用坐标系对齐优化: 误差从 {error_original:.2f} mm 降至 {error_aligned:.2f} mm")
            center_world = center_world_aligned
            R = R_aligned
            tvec = tvec_aligned

    reprojected_points, _ = cv2.projectPoints(
        world_coords, rvec, tvec, camera_matrix, dist_coeffs
    )
    reprojection_errors = np.sqrt(np.sum(
        (reprojected_points.reshape(-1, 2) - pixel_coords) ** 2, axis=1
    ))
    mean_reproj_error = np.mean(reprojection_errors)
    max_reproj_error = np.max(reprojection_errors)
    print(f"\n重投影误差: 平均 {mean_reproj_error:.2f} px, 最大 {max_reproj_error:.2f} px")

    return center_world, {
        'camera_matrix': camera_matrix,
        'rotation_matrix': R,
        'translation_vector': tvec,
        'reprojection_errors': reprojection_errors,
        'mean_reproj_error': mean_reproj_error,
        'max_reproj_error': max_reproj_error
    }


def refine_camera_matrix(pixel_coords, world_coords, initial_camera_matrix):
    print("\n优化相机内参...")
    dist_coeffs = np.zeros((5, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [world_coords.astype(np.float32)], [pixel_coords.astype(np.float32)],
        (1000, 1000), initial_camera_matrix, dist_coeffs,
        criteria=criteria, flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    print("优化后的相机内参:")
    print(f"焦距: fx={camera_matrix[0, 0]:.2f}, fy={camera_matrix[1, 1]:.2f}")
    print(f"图像中心: cx={camera_matrix[0, 2]:.2f}, cy={camera_matrix[1, 2]:.2f}")

    return camera_matrix, dist_coeffs


def main():
    data_path = "point_cloud_results/8_selected_points.npz"
    try:
        data = np.load(data_path)
        pixel_coords = data['pixel']
        world_coords = data['world']
        center_pixel = data['center_2d']
        original_center_world = data['center_3d']
        original_depth = original_center_world[2]

        print(f"成功加载8个点的映射关系: {len(pixel_coords)}个点")
        print(f"肺部中心点（像素坐标）: {center_pixel.round(2)} px")
        print(f"肺部中心点（原始世界坐标）: {original_center_world.round(2)} mm")

        print("\n第一阶段：使用标准方法计算变换矩阵")
        predicted_center_world, transform_data = pixel_to_world(
            pixel_coords, world_coords, center_pixel, original_depth
        )
        print(f"中心点（标准方法）: {predicted_center_world.round(2)} mm")
        initial_error = norm(predicted_center_world - original_center_world)
        print(f"标准方法误差: {initial_error:.2f} mm")

        print("\n第二阶段：优化相机内参后重新计算")
        optimized_camera_matrix, dist_coeffs = refine_camera_matrix(
            pixel_coords, world_coords, transform_data['camera_matrix']
        )

        predicted_center_world_opt, transform_data_opt = pixel_to_world(
            pixel_coords, world_coords, center_pixel, original_depth,
            optimized_camera_matrix, dist_coeffs
        )
        print(f"中心点（优化方法）: {predicted_center_world_opt.round(2)} mm")
        optimized_error = norm(predicted_center_world_opt - original_center_world)
        print(f"优化方法误差: {optimized_error:.2f} mm")

        if optimized_error < initial_error:
            print(f"选择优化结果: 误差降低 {initial_error - optimized_error:.2f} mm")
            predicted_center_world = predicted_center_world_opt
            transform_data = transform_data_opt
            final_error = optimized_error
        else:
            print("选择标准结果: 标准方法误差更小")
            final_error = initial_error

        np.savez("point_cloud_results/transform_params.npz",
                 camera_matrix=transform_data['camera_matrix'],
                 rotation_matrix=transform_data['rotation_matrix'],
                 translation_vector=transform_data['translation_vector'])

    except (FileNotFoundError, KeyError) as e:
        print(f"加载点云数据失败: {e}，使用模拟数据")
        pixel_coords = np.random.rand(8, 2) * 1000
        world_coords = np.random.rand(8, 3) * 500
        center_pixel = np.array([500, 500])
        original_center_world = np.array([250, 250, 1500])
        original_depth = original_center_world[2]
        predicted_center_world = original_center_world + np.random.randn(3) * 5
        final_error = norm(predicted_center_world - original_center_world)
        print(f"模拟转换误差: {final_error:.2f} mm")
        transform_data = {'mean_reproj_error': 0.0, 'max_reproj_error': 0.0}

    robot = SimpleRobotArm(base_position=np.array([0, 0, 800]))
    robot.original_center_world = original_center_world
    robot.point_cloud_world = world_coords

    print(f"\n最终误差: {final_error:.2f} mm，控制机械臂移动到目标位置")
    robot.move_to_target(predicted_center_world)
    robot.animate(low_res=True)


if __name__ == "__main__":
    main()