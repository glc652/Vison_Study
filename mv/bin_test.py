import cv2
import matplotlib.pyplot as plt
import numpy as np

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 黑体、微软雅黑、宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# img_row = cv2.imread(r"D:\vision\mv\2.png")
# kernel = np.ones((38,38),np.uint8)
# tophat = cv2.morphologyEx(img_row, cv2.MORPH_TOPHAT, kernel)

# img_gray = cv2.cvtColor(tophat, cv2.COLOR_BGR2GRAY)
# img_gray2 = cv2.cvtColor(img_row, cv2.COLOR_BGR2GRAY)

# ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

# # 大津法（Otsu's method）
# ret_otsu, thresh6 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret_otsu, thresh7 = cv2.threshold(img_gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# title = ['Original Image','tophat' ,'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'OTSU','OTSU2']
# images = [img_row, tophat,thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7]
# for i in range(9):
#     plt.subplot(2, 5, i+1), plt.imshow(images[i], 'gray')
#     plt.title(title[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_row = cv2.imread(r"D:\vision\mv\1.png")

img_gray = cv2.cvtColor(img_row, cv2.COLOR_BGR2GRAY)

# 1. 中值滤波去除噪点
denoised = cv2.medianBlur(img_gray, 3)

# # 2. CLAHE 增强对比度（clipLimit 适当增大）
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# enhanced_gray = clahe.apply(denoised)

# 3. Otsu 二值化
ret_otsu, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. 形态学开运算和闭运算对比
kernel_clean = np.ones((3,3),np.uint8)

# 开运算：去除小白点（前景噪点）
open_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean, iterations=1)

# 闭运算：去除小黑点（背景噪点/空洞）
close_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clean, iterations=1)

# 并排显示对比
titles = ['Original', 'Open(开运算)', 'Close(闭运算)']
images = [thresh, open_thresh, close_thresh]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
