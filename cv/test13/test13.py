import cv2
import numpy as np
import matplotlib.pyplot as plt

def laplacian_sharpen(img_gray):
    # 1. 定义4邻域拉普拉斯卷积核
    lap_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # 2. 卷积得到拉普拉斯二阶微分图
    lap = cv2.filter2D(img_gray, cv2.CV_32F, lap_kernel)
    # 归一化拉普拉斯图方便展示
    lap_show = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 3. 锐化：原图 - 拉普拉斯分量
    sharp = img_gray.astype(np.float32) - lap
    # 裁剪像素值到0~255，防止溢出
    sharp = np.clip(sharp, 0, 255)
    sharp = sharp.astype(np.uint8)

    return sharp, lap_show

if __name__ == "__main__":
    # 图像路径
    image_path = r"D:\vision\cv\images\Fig0301.png"
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError("路径错误，图片未找到：" + image_path)

    # 拉普拉斯锐化
    img_sharp, lap_img = laplacian_sharpen(img)

    # 绘图展示
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("原始灰度图像")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(lap_img, cmap="gray")
    plt.title("拉普拉斯二阶微分图")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_sharp, cmap="gray")
    plt.title("拉普拉斯锐化结果")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 保存锐化后的图像
    cv2.imwrite("laplacian_sharp_Fig0301.png", img_sharp)