import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def load_image(path):
    """加载图像并转为灰度"""
    img = Image.open(path).convert('L')  # 转为灰度图
    return np.array(img)

def sobel_operator(image):
    """
    手动实现 Sobel 算子
    """
    # Sobel 核
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    h, w = image.shape
    output_x = np.zeros((h, w), dtype=np.float64)
    output_y = np.zeros((h, w), dtype=np.float64)

    # 填充边界（补零）
    padded = np.pad(image.astype(np.float64), pad_width=1, mode='constant', constant_values=0)

    # 卷积操作
    for i in range(h):
        for j in range(w):
            # 提取 3x3 区域
            region = padded[i:i+3, j:j+3]
            # 卷积计算
            output_x[i, j] = np.sum(region * sobel_x)
            output_y[i, j] = np.sum(region * sobel_y)

    return output_x, output_y

def normalize(image):
    """归一化到 0-255"""
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

def combine_edges(grad_x, grad_y):
    """组合两个方向的梯度"""
    # 计算梯度幅值
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return normalize(magnitude)

# 主程序
if __name__ == "__main__":
    # 读取图像
    img_path = r"D:\vision\mv\1.png"
    gray = load_image(img_path)

    # 应用 Sobel 算子
    grad_x, grad_y = sobel_operator(gray)

    # 归一化显示
    grad_x_norm = normalize(grad_x)
    grad_y_norm = normalize(grad_y)

    # 组合边缘
    combined = combine_edges(grad_x, grad_y)

    # 显示结果
    titles = ['原始图像', 'Sobel X (垂直边缘)', 'Sobel Y (水平边缘)', 'Sobel 组合']
    images = [gray, grad_x_norm, grad_y_norm, combined]

    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

    print("Sobel 算子完成！")
    print(f"梯度 X 范围：{grad_x.min():.2f} ~ {grad_x.max():.2f}")
    print(f"梯度 Y 范围：{grad_y.min():.2f} ~ {grad_y.max():.2f}")
