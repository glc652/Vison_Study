# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # 读取彩色图像并转换为灰度图像
# image_path = r"D:\vision\cv\images\flowers.tif"
# color_img = cv2.imread(image_path)
# gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

# # 计算灰度直方图（频数）
# hist_freq = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
# # 计算相对频数（归一化直方图）
# hist_normalized = hist_freq.ravel() / hist_freq.sum()
# # 计算累积直方图（相对频数的累积和）
# cumulative_hist = np.cumsum(hist_normalized)

# fig, axs = plt.subplots(1, 3, figsize=(15, 4))
# # 灰度图像
# axs[0].imshow(gray_img, cmap='gray')
# axs[0].set_title('Gray Image')
# axs[0].axis('off')
# # 灰度值频数直方图
# axs[1].bar(range(256), hist_freq.ravel(), width=1.0, color='black')
# axs[1].set_title('Grayscale Frequency Histogram')
# axs[1].set_xlabel('Pixel Intensity')
# axs[1].set_ylabel('Frequency')
# # 灰度值相对频数累积直方图
# axs[2].plot(cumulative_hist, color='black')
# axs[2].set_title('Cumulative Relative Frequency Histogram')
# axs[2].set_xlabel('Pixel Intensity')
# axs[2].set_ylabel('Cumulative Relative Frequency')
# axs[2].set_ylim([0, 1])
# plt.tight_layout()
# plt.show()
# 显示彩色图像及其各通道直方图
# fig2, ax = plt.subplots(figsize=(10, 4))
# colors = ('b', 'g', 'r')
# channel_names = ('Blue', 'Green', 'Red')

# for i, col in enumerate(colors):
#     hist = cv2.calcHist([color_img], [i], None, [256], [0, 256])
#     ax.plot(hist, color=col, label=channel_names[i])

# ax.set_title('Color Image Histogram (BGR Channels)')
# ax.set_xlabel('Pixel Intensity')
# ax.set_ylabel('Frequency')
# ax.legend()
# ax.set_xlim([0, 256])

# plt.tight_layout()
# plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r"D:\vision\cv\images\flowers.tif"
color_img = cv2.imread(image_path)
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray_img)  
hist_orig = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
hist_clahe = cv2.calcHist([gray_clahe], [0], None, [256], [0, 256])

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# 原始灰度图像
axs[0, 0].imshow(gray_img, cmap='gray')
axs[0, 0].set_title('Original Gray Image')
axs[0, 0].axis('off')
# CLAHE 均衡化后的图像
axs[0, 1].imshow(gray_clahe, cmap='gray')
axs[0, 1].set_title('CLAHE Enhanced Image')
axs[0, 1].axis('off')
# 原始直方图
axs[1, 0].plot(hist_orig, color='black')
axs[1, 0].set_title('Histogram (Original)')
axs[1, 0].set_xlabel('Pixel Intensity')
axs[1, 0].set_ylabel('Frequency')
# CLAHE 后的直方图
axs[1, 1].plot(hist_clahe, color='black')
axs[1, 1].set_title('Histogram (CLAHE)')
axs[1, 1].set_xlabel('Pixel Intensity')
axs[1, 1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()