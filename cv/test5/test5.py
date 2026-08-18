# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# def plot_grayHist(img, rows, cols, idx):
#     plt.subplot(rows, cols, idx)
#     histogram, bins, patch = plt.hist(img.ravel(), 256, histtype='bar', density=True)   
#     plt.xlabel('灰度级')
#     plt.ylabel('像素比例')
#     plt.axis([0, 256, 0, np.max(histogram)])

# def task1_linear_stretch(img):
#     min_val, max_val = img.min(), img.max()
#     if max_val == min_val:
#         out_img1 = img.copy()
#     else:
#         out_img1 = np.clip((img - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)

#     p1, p99 = np.percentile(img, (1, 99))
#     if p99 == p1:
#         out_img2 = img.copy()
#     else:
#         out_img2 = np.clip((img - p1) * 255.0 / (p99 - p1), 0, 255).astype(np.uint8)

#     return img, out_img1, out_img2

# if __name__ == "__main__":
#     img_path = r"D:\vision\cv\images\Fig0703.png"
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"无法读取图像: {img_path}")

#     orig, res1, res2 = task1_linear_stretch(img)

#     plt.figure(figsize=(14, 6))

#     plt.subplot(2, 3, 1)
#     plt.imshow(orig, cmap='gray')
#     plt.title('原始图像')
#     plt.axis('off')  

#     plt.subplot(2, 3, 4)
#     plot_grayHist(orig, 2, 3, 4)

#     plt.subplot(2, 3, 2)
#     plt.imshow(res1, cmap='gray')
#     plt.title('线性变换（最小/最大值）')
#     plt.axis('off')

#     plt.subplot(2, 3, 5)
#     plot_grayHist(res1, 2, 3, 5)

#     plt.subplot(2, 3, 3)
#     plt.imshow(res2, cmap='gray')
#     plt.title('线性变换（1%~99%百分位）')
#     plt.axis('off')

#     plt.subplot(2, 3, 6)
#     plot_grayHist(res2, 2, 3, 6)

#     plt.tight_layout()
#     plt.show()

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # === 解决中文乱码 ===
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# def plot_grayHist(img, rows, cols, idx):
#     plt.subplot(rows, cols, idx)
#     histogram, bins, patch = plt.hist(img.ravel(), 256, histtype='bar', density=True)   
#     plt.xlabel('灰度级')
#     plt.ylabel('像素比例')
#     plt.axis([0, 256, 0, np.max(histogram)])

# def task2_piecewise_linear(img, r1=70, s1=30, r2=140, s2=200):
#     """
#     分段线性变换：
#     - [0, r1]   → [0, s1]
#     - (r1, r2)  → (s1, s2)
#     - [r2, 255] → [s2, 255]
#     """
#     if r1 >= r2:
#         r1, r2 = min(r1, r2), max(r1, r2)
    
#     out = np.zeros_like(img, dtype=np.float32)
    
#     mask1 = img <= r1
#     out[mask1] = (s1 / (r1 + 1e-8)) * img[mask1] 
#     mask2 = (img > r1) & (img < r2)
#     out[mask2] = ((s2 - s1) / (r2 - r1 + 1e-8)) * (img[mask2] - r1) + s1
#     mask3 = img >= r2
#     out[mask3] = ((255 - s2) / (255 - r2 + 1e-8)) * (img[mask3] - r2) + s2
    
#     return np.clip(out, 0, 255).astype(np.uint8)

# if __name__ == "__main__":
#     img_path = r"D:\vision\cv\images\Fig0703.png"
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"无法读取图像: {img_path}")

#     r1, s1 = 70, 30
#     r2, s2 = 140, 200
#     enhanced_img = task2_piecewise_linear(img, r1=r1, s1=s1, r2=r2, s2=s2)
#     plt.figure(figsize=(10, 6))
#     plt.subplot(2, 2, 1)
#     plt.imshow(img, cmap='gray')
#     plt.title('原始图像')
#     plt.axis('off')
#     plt.subplot(2, 2, 3)
#     plot_grayHist(img, 2, 2, 3)
#     plt.subplot(2, 2, 2)
#     plt.imshow(enhanced_img, cmap='gray')
#     plt.title(f'分段线性增强\n(r1={r1},s1={s1}; r2={r2},s2={s2})')
#     plt.axis('off')
#     plt.subplot(2, 2, 4)
#     plot_grayHist(enhanced_img, 2, 2, 4)
#     plt.tight_layout()
#     plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_grayHist(img, rows, cols, idx):
    plt.subplot(rows, cols, idx)
    histogram, bins, patch = plt.hist(img.ravel(), 256, histtype='bar', density=True)   
    plt.xlabel('灰度级')
    plt.ylabel('像素比例')
    plt.axis([0, 256, 0, np.max(histogram)])

def task3_brightness_contrast(img, alpha=1.0, beta=0):
    """
    调整图像亮度和对比度。
    :param img: 输入灰度图像 (uint8)
    :param alpha: 对比度增益 (默认1.0，无变化)
    :param beta: 亮度偏移 (默认0，无变化)
    :return: 调整后的图像 (uint8)
    """
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

if __name__ == "__main__":
    img_path = r"D:\vision\cv\images\Fig0703.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")

    alpha = 1.5   # 对比度增益（>1 增强对比度）
    beta = 20     # 亮度偏移（>0 变亮）

    adjusted_img = task3_brightness_contrast(img, alpha=alpha, beta=beta)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plot_grayHist(img, 2, 2, 3)

    plt.subplot(2, 2, 2)
    plt.imshow(adjusted_img, cmap='gray')
    plt.title(f'亮度/对比度调整\n(α={alpha}, β={beta})')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plot_grayHist(adjusted_img, 2, 2, 4)

    plt.tight_layout()
    plt.show()