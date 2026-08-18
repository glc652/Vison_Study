import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_calibration_result(calib_file):
    """从标定结果文件中加载相机内参和畸变系数"""
    try:
        with open(calib_file, 'r', encoding='gbk') as f:
            lines = f.readlines()
    except:
        with open(calib_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

    A = None
    k = None

    # 查找内参矩阵
    for i, line in enumerate(lines):
        if '[[' in line and i > 0:
            try:
                matrix_lines = []
                j = i
                while j < len(lines) and ']]' not in lines[j]:
                    matrix_lines.append(lines[j])
                    j += 1
                if j < len(lines):
                    matrix_lines.append(lines[j])

                matrix_str = ''.join(matrix_lines)
                matrix_str = matrix_str.replace('[[', '').replace(']]', '')
                matrix_str = matrix_str.replace('[', '').replace(']', '')

                rows = []
                for line in matrix_str.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            row = [float(x) for x in line.split()]
                            if len(row) == 3:
                                rows.append(row)
                        except:
                            pass

                if len(rows) == 3:
                    A = np.array(rows)
                    break
            except:
                pass

    # 查找畸变系数 - 支持两种格式
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # 格式1: [[-0.43... 0.48... ...]]
        if '[[' in line_stripped and '-' in line_stripped:
            try:
                dist_str = line_stripped.replace('[[', '').replace(']]', '').replace('[', '').replace(']', '')
                k_list = [float(x) for x in dist_str.split()]
                if len(k_list) >= 2:
                    k = np.array(k_list)
                    break
            except:
                pass
        # 格式2: [-0.43... 0.48... ...] (单括号)
        elif line_stripped.startswith('[') and line_stripped.endswith(']') and '-' in line_stripped:
            try:
                dist_str = line_stripped.replace('[', '').replace(']', '')
                k_list = [float(x) for x in dist_str.split()]
                if len(k_list) >= 2:
                    k = np.array(k_list)
                    break
            except:
                pass

    return A, k


def undistort_image(img, A, k):
    """对图像进行畸变校正"""
    h, w = img.shape[:2]
    k_full = np.concatenate([k, np.zeros(3)])
    map_x, map_y = cv2.initUndistortRectifyMap(A, k_full, None, A, (w, h), 5)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


def create_comparison_figure(img_orig, img_opencv, img_zhang, filename, output_dir):
    """创建对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_orig, cmap='gray')
    axes[0].set_title('原始图像', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img_opencv, cmap='gray')
    axes[1].set_title('OpenCV 校正', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(img_zhang, cmap='gray')
    axes[2].set_title('Zhang 校正', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    # 保存图像
    output_file = os.path.join(output_dir, f"{filename[:-4]}_comparison.png")
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()

    return output_file


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

    # 检查数据有效性
    if A_opencv is None or k_opencv is None:
        print("OpenCV 标定结果加载失败")
        return

    # 如果 Zhang 加载失败，使用 OpenCV 结果
    if A_zhang is None or k_zhang is None:
        print("Zhang 标定结果加载失败，使用 OpenCV 结果代替")
        A_zhang = A_opencv
        k_zhang = k_opencv

    # 确保畸变系数是 1D 数组
    if k_opencv.ndim > 1:
        k_opencv = k_opencv.flatten()
    if k_zhang.ndim > 1:
        k_zhang = k_zhang.flatten()

    # 获取测试图像列表
    test_files = [f for f in os.listdir(test_dir)
                  if f.lower().endswith(('.bmp', '.jpg', '.png'))]
    test_files.sort()

    if not test_files:
        print(f"未找到测试图像在 {test_dir}")
        return

    print(f"找到 {len(test_files)} 张测试图像\n")

    # 处理每张测试图像
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        img = cv2.imread(test_path, 0)

        if img is None:
            print(f"无法读取 {test_file}")
            continue

        print(f"处理: {test_file}")

        # 进行畸变校正
        img_opencv = undistort_image(img, A_opencv, k_opencv)
        img_zhang = undistort_image(img, A_zhang, k_zhang)

        # 生成对比图
        output_file = create_comparison_figure(img, img_opencv, img_zhang, test_file, output_dir)
        print(f"  已保存: {output_file}\n")

    print(f"所有结果已保存到 {output_dir}")


if __name__ == '__main__':
    main()
