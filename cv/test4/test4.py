import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = r"D:\vision\cv\images\Fig1701.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# # 方法1：使用 OpenCV 的 convertScaleAbs
# brighter_cv = cv2.convertScaleAbs(img, alpha=1, beta=100)   # 增加亮度100
# darker_cv = cv2.convertScaleAbs(img, alpha=1, beta=-75)     # 降低亮度75

# # 方法2：使用 NumPy 直接加减
# brighter_np = np.clip(img.astype(np.int16) + 100, 0, 255).astype(np.uint8)
# darker_np = np.clip(img.astype(np.int16) - 75, 0, 255).astype(np.uint8)

# fig, axes = plt.subplots(2, 3, figsize=(12, 8))
# axes[0, 0].imshow(img, cmap='gray')
# axes[0, 0].set_title('original')
# axes[0, 1].imshow(brighter_cv, cmap='gray')
# axes[0, 1].set_title('OpenCV +100')
# axes[0, 2].imshow(darker_cv, cmap='gray')
# axes[0, 2].set_title('OpenCV -75')

# axes[1, 0].imshow(img, cmap='gray')
# axes[1, 0].set_title('original')
# axes[1, 1].imshow(brighter_np, cmap='gray')
# axes[1, 1].set_title('NumPy +100')
# axes[1, 2].imshow(darker_np, cmap='gray')
# axes[1, 2].set_title('NumPy -75')
# for ax in axes.flat:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()
# print("增加亮度结果一致？", np.array_equal(brighter_cv, brighter_np))
# print("降低亮度结果一致？", np.array_equal(darker_cv, darker_np))

## 方法1：使用 OpenCV 的 convertScaleAbs 
# high_contrast_cv = cv2.convertScaleAbs(img, alpha=1.5, beta=0)   # 对比度 ×1.5
# low_contrast_cv = cv2.convertScaleAbs(img, alpha=0.5, beta=0)    # 对比度 ×0.5

# # 方法2：使用 NumPy 
# high_contrast_np = np.clip(img.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
# low_contrast_np = np.clip(img.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
# fig, axes = plt.subplots(2, 3, figsize=(12, 8))
# # 第一行：OpenCV 结果
# axes[0, 0].imshow(img, cmap='gray')
# axes[0, 0].set_title('original')
# axes[0, 1].imshow(high_contrast_cv, cmap='gray')
# axes[0, 1].set_title('OpenCV ×1.5')
# axes[0, 2].imshow(low_contrast_cv, cmap='gray')
# axes[0, 2].set_title('OpenCV ×0.5')
# # 第二行：NumPy 结果
# axes[1, 0].imshow(img, cmap='gray')
# axes[1, 0].set_title('original')
# axes[1, 1].imshow(high_contrast_np, cmap='gray')
# axes[1, 1].set_title('NumPy ×1.5')
# axes[1, 2].imshow(low_contrast_np, cmap='gray')
# axes[1, 2].set_title('NumPy ×0.5')
# for ax in axes.flat:
#     ax.axis('off')

# plt.tight_layout()
# plt.show()
# print("高对比度结果一致？", np.array_equal(high_contrast_cv, high_contrast_np))
# print("低对比度结果一致？", np.array_equal(low_contrast_cv, low_contrast_np))
# 对比度调整（保留你已有的代码）
high_contrast_cv = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
low_contrast_cv = cv2.convertScaleAbs(img, alpha=0.5, beta=0)

# 使用 OpenCV 计算直方图（bins=256, 范围[0,256)）
hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_high = cv2.calcHist([high_contrast_cv], [0], None, [256], [0, 256])
hist_low = cv2.calcHist([low_contrast_cv], [0], None, [256], [0, 256])

# 创建图像和直方图的综合显示
fig, axes = plt.subplots(2, 3, figsize=(14, 10))

# 第一行：图像显示
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(high_contrast_cv, cmap='gray')
axes[0, 1].set_title('High Contrast (×1.5)')
axes[0, 2].imshow(low_contrast_cv, cmap='gray')
axes[0, 2].set_title('Low Contrast (×0.5)')

# 第二行：直方图显示
axes[1, 0].plot(hist_img, color='black')
axes[1, 0].set_title('Histogram - Original')
axes[1, 0].set_xlim([0, 256])

axes[1, 1].plot(hist_high, color='black')
axes[1, 1].set_title('Histogram - High Contrast')
axes[1, 1].set_xlim([0, 256])

axes[1, 2].plot(hist_low, color='black')
axes[1, 2].set_title('Histogram - Low Contrast')
axes[1, 2].set_xlim([0, 256])
for ax in axes.flat:
    ax.axis('on')  
plt.tight_layout()
plt.show()

