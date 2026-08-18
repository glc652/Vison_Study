import os
import cv2
import numpy as np
from srcs.utils import save_result


def load_calibration_result(calib_file):
    """从标定结果文件中加载相机内参和畸变系数"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    content = None

    for encoding in encodings:
        try:
            with open(calib_file, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if content is None:
        raise ValueError(f"无法读取文件 {calib_file}，尝试了多种编码")

    # 解析文件内容
    A = None
    k = None

    # 查找相机内参部分
    if '相机内参' in content or '内参' in content:
        # 提取内参矩阵（3x3）
        start_idx = content.find('相机内参')
        if start_idx == -1:
            start_idx = content.find('内参')

        # 从标记后开始查找矩阵数据
        matrix_start = content.find('[[', start_idx)
        if matrix_start == -1:
            matrix_start = content.find('[', start_idx)

        matrix_end = content.find(']]', matrix_start) + 2
        if matrix_end == 1:  # 没找到]]
            matrix_end = content.find(']', matrix_start) + 1

        matrix_str = content[matrix_start:matrix_end]
        # 移除所有括号和多余空白
        matrix_str = matrix_str.replace('[', '').replace(']', '')
        # 按行分割并解析
        rows = []
        for line in matrix_str.strip().split('\n'):
            line = line.strip()
            if line:
                row = [float(x) for x in line.split()]
                if row:
                    rows.append(row)
        if rows:
            A = np.array(rows)

    # 查找相机畸变部分
    if '相机畸变' in content or '畸变' in content:
        start_idx = content.find('相机畸变')
        if start_idx == -1:
            start_idx = content.find('畸变')

        # 从标记后开始查找数组数据
        array_start = content.find('[', start_idx)
        array_end = content.find(']', array_start) + 1

        array_str = content[array_start:array_end]
        # 移除所有括号
        array_str = array_str.replace('[', '').replace(']', '')
        # 解析数值
        values = [float(x) for x in array_str.split() if x.strip()]
        k = np.array(values)

    return A, k


def test_undistortion(test_dir, output_dir, A_opencv, k_opencv, A_zhang, k_zhang):
    """对测试图像进行畸变校正"""

    # 获取测试图像列表
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

        # 保存结果
        output_file_orig = os.path.join(output_dir, f"{filename[:-4]}_original.bmp")
        output_file_opencv = os.path.join(output_dir, f"{filename[:-4]}_opencv.bmp")
        output_file_zhang = os.path.join(output_dir, f"{filename[:-4]}_zhang.bmp")

        cv2.imwrite(output_file_orig, img)
        cv2.imwrite(output_file_opencv, img_opencv)
        cv2.imwrite(output_file_zhang, img_zhang)

        print(f"已处理: {filename}")
        print(f"  原始: {output_file_orig}")
        print(f"  OpenCV: {output_file_opencv}")
        print(f"  Zhang: {output_file_zhang}")


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

    # 对测试图像进行畸变校正
    print(f"处理测试图像...")
    test_undistortion(test_dir, output_dir, A_opencv, k_opencv, A_zhang, k_zhang)

    print(f"\n结果已保存到 {output_dir}")


if __name__ == '__main__':
    main()
