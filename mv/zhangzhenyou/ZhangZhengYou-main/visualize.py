import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from srcs.utils import save_result


def load_calibration_result(calib_file):
    """从标定结果文件中加载相机内参和畸变系数"""
    with open(calib_file, 'r') as f:
        lines = f.readlines()

    A = None
    k = None

    for i, line in enumerate(lines):
        if '相机内参' in line:
            A = []
            for j in range(3):
                row_str = lines[i+j+1].strip()
                if row_str.startswith('['):
                    row_str = row_str[1:]
                if row_str.endswith(']'):
                    row_str = row_str[:-1]
                row = [float(x) for x in row_str.split()]
                A.append(row)
            A = np.array(A)

        if '相机畸变' in line:
            dist_str = lines[i+1].strip()
            if dist_str.startswith('['):
                dist_str = dist_str[1:]
            if dist_str.endswith(']'):
                dist_str = dist_str[:-1]
            k = np.array([float(x) for x in dist_str.split()])

    return A, k


def visualize_undistortion(test_dir, output_dir, A_opencv, k_opencv, A_zhang, k_zhang):
    """生成原始、OpenCV校正、Zhang校正的拼接图"""

    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                  if f.lower().endswith(('.bmp', '.jpg', '.png'))]
    test_files.sort()

    if not test_files:
        print(f"未找到测试图像在 {test_dir}")
        return

    print(f"找到 {len(test_files)} 张测试图像")

    for test_file in test_files:
        img = cv2.imread(test_file, 0)
        if img is None:
            print(f"无法读取 {test_file}")
            continue

        h, w = img.shape[:2]
        filename = os.path.basename(test_file)

        # OpenCV 畸变校正
        k_opencv_full = np.concatenate([k_opencv, np.zeros(3)])
        map_opencv_x, map_opencv_y = cv2.initUndistortRectifyMap(
            A_opencv, k_opencv_full, None, A_opencv, (w, h), 5)
        img_opencv = cv2.remap(img, map_opencv_x, map_opencv_y, cv2.INTER_LINEAR)

        # Zhang 畸变校正
        k_zhang_full = np.concatenate([k_zhang, np.zeros(3)])
        map_zhang_x, map_zhang_y = cv2.initUndistortRectifyMap(
            A_zhang, k_zhang_full, None, A_zhang, (w, h), 5)
        img_zhang = cv2.remap(img, map_zhang_x, map_zhang_y, cv2.INTER_LINEAR)

        # 生成拼接图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(img_opencv, cmap='gray')
        axes[1].set_title('OpenCV Undistorted')
        axes[1].axis('off')

        axes[2].imshow(img_zhang, cmap='gray')
        axes[2].set_title('Zhang Undistorted')
        axes[2].axis('off')

        plt.tight_layout()

        # 保存拼接图
        output_file = os.path.join(output_dir, f"{filename[:-4]}_comparison.png")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"已生成拼接图: {output_file}")


def main():
    # 路径配置
    calib_dir = r"D:\vision\mv\zhangzhenyou\output"
    test_dir = r"D:\vision\mv\zhangzhenyou\used data\test"
    output_dir = r"D:\vision\mv\zhangzhenyou\output\test_results"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载标定结果
    calib_file_opencv = os.path.join(calib_dir, "opencv.txt")
    calib_file_zhang = os.path.join(calib_dir, "zhang.txt")

    print("加载标定结果...")
    A_opencv, k_opencv = load_calibration_result(calib_file_opencv)
    A_zhang, k_zhang = load_calibration_result(calib_file_zhang)

    print(f"OpenCV 内参:\n{A_opencv}")
    print(f"OpenCV 畸变: {k_opencv}\n")
    print(f"Zhang 内参:\n{A_zhang}")
    print(f"Zhang 畸变: {k_zhang}\n")

    # 生成拼接图
    print(f"生成拼接图...")
    visualize_undistortion(test_dir, output_dir, A_opencv, k_opencv, A_zhang, k_zhang)

    print(f"\n拼接图已保存到 {output_dir}")


if __name__ == '__main__':
    main()
