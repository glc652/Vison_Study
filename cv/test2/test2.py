import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
# ====题目一
# 导入库，读入彩色图像并复制数组，查看图像维度、形状与数据类型。
# img = cv.imread("D:\\vision\\cv\\pic\\flowers.tif",cv.IMREAD_COLOR)
# img2 = img.copy()
# print('数组维数：',img.ndim)
# print('图像大小：',img.shape)
# print('图像数据类型类型：',img.dtype)

# # 读取指定坐标像素 BGR 值，修改像素值、画线并复制图像子区域
# pixb = img[100,300,0]
# print('坐标[100,300]处像素b分量值:',pixb)
# pixbgr = img[100,300,:]
# print('坐标[100,300]处像素bgr分量值:',pixbgr)
# img2[100,300,1] = 200
# img2[300:302,200:300,:] = [0,0,255]
# nameplate = img2[197:257,390:490,:]
# img2[200:260,266:366,:] = nameplate

# # 对比显示原始与修改后图像，
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.imshow(img[:,:,::-1])
# plt.title('oled_villa,Original image')
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.imshow(img2[:,:,::-1])
# plt.title('oled_villa,Some pixels changed')
# plt.axis('off')
# plt.show()

# # 分离并显示 RGB 各颜色分量。
# imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
# plt.figure(figsize=(12,8))
# plt.gray()
# plt.subplot(2, 3, 1)
# plt.imshow(imgRGB[:,:,0])
# plt.title('Color component R')
# plt.axis('off')
# plt.subplot(2, 3, 2)
# plt.imshow(imgRGB[:,:,1])
# plt.title('Color component G')
# plt.axis('off')
# plt.subplot(2, 3, 3)
# plt.imshow(imgRGB[:,:,2])
# plt.title('Color component B')
# plt.axis('off')

# imgR = imgRGB.copy()
# imgR[:,:,1:3] = 0
# imgG = imgRGB.copy()
# imgG[:,:,0] = 0
# imgG[:,:,2] = 0
# imgB = imgRGB.copy()
# imgB[:,:,0:2] = 0
# plt.subplot(2, 3, 4)
# plt.imshow(imgR)
# plt.axis('off')
# plt.subplot(2, 3, 5)
# plt.imshow(imgG)
# plt.axis('off')
# plt.subplot(2, 3, 6)
# plt.imshow(imgB)
# plt.axis('off')
# plt.show()

# ===题目二
# 分离彩色图像 RGB 通道并单独显示，将分离通道重新合并，验证通道操作的正确性。
# img = cv.imread("D:\\vision\\cv\\pic\\flowers.tif", cv.IMREAD_COLOR)
# imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# # 分离 RGB 
# R, G, B = cv.split(imgRGB)

# plt.figure(figsize=(15, 10))
# plt.subplot(2, 3, 1)
# plt.imshow(R, cmap='gray')
# plt.title('R Channel')
# plt.axis('off')

# plt.subplot(2, 3, 2)
# plt.imshow(G, cmap='gray')
# plt.title('G Channel')
# plt.axis('off')

# plt.subplot(2, 3, 3)
# plt.imshow(B, cmap='gray')
# plt.title('B Channel')
# plt.axis('off')

# imgMerged = cv.merge([R, G, B])

# plt.subplot(2, 3, 4)
# plt.imshow(imgMerged)
# plt.title('Merged Image')
# plt.axis('off')

# plt.subplot(2, 3, 5)
# plt.imshow(imgRGB)
# plt.title('Original RGB Image')
# plt.axis('off')

# # 验证通道操作的正确性：计算原始图像与合并图像的差值
# diff = cv.absdiff(imgRGB, imgMerged)
# diff_sum = np.sum(diff)
# print(f'原始图像与合并图像的差异值总和：{diff_sum}')
# if diff_sum == 0:
#     print('验证成功：通道分离与合并操作完全正确！')
# else:
#     print('验证失败：通道操作存在误差')
# plt.subplot(2, 3, 6)
# plt.imshow(imgRGB)
# plt.title(f'Verification (diff={diff_sum})')
# plt.axis('off')
# plt.tight_layout()
# plt.show()

# ===题目三
# 读取索引图像，解析调色板，将索引值映射为 RGB 值，生成真彩色图像并显示对比。
img_indexed = Image.open("D:\\vision\\cv\\pic\\flowers.tif")  # 替换为实际的索引图像文件

# 检查是否为索引图像
if img_indexed.mode != 'P':
    print('警告：图像不是索引模式，尝试转换为索引模式')
    img_indexed = img_indexed.convert('P')

# 获取调色板数据
palette = img_indexed.getpalette()  # 返回长度为 768 的列表 (256*3, RGB 各 256 个值)
print(f'调色板长度：{len(palette)}')

# 获取图像的索引数据
img_array = np.array(img_indexed)
print(f'索引图像形状：{img_array.shape}, 数据类型：{img_array.dtype}')
print(f'唯一索引值数量：{len(np.unique(img_array))}')

# 将调色板转换为 numpy 数组并重塑为 (256, 3) 形式
palette_np = np.array(palette, dtype=np.uint8).reshape(-1, 3)

# 将索引值映射为 RGB 值
img_truecolor = palette_np[img_array]
print(f'真彩色图像形状：{img_truecolor.shape}, 数据类型：{img_truecolor.dtype}')

# 显示对比
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_indexed)
plt.title('Indexed Image (PIL auto-convert)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_array, cmap='gray')
plt.title(f'Index Map (shape={img_array.shape})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_truecolor)
plt.title('True Color Image (from palette mapping)')
plt.axis('off')
plt.tight_layout()
plt.show()
