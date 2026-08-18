import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = cv2.imread(r"D:\vision\cv\images\Lena.tif", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("无法读取图像")
kernel_size = 15
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
M, N = img.shape
kernel_padded = np.zeros((M, N), dtype=np.float32)
kernel_padded[:kernel_size, :kernel_size] = kernel
kernel_centered = np.roll(kernel_padded, shift=(-kernel_size//2, -kernel_size//2), axis=(0,1))
F = np.fft.fft2(img.astype(np.float32))
H = np.fft.fft2(kernel_centered)
G = F * H
g = np.fft.ifft2(G).real
g = np.clip(g, 0, 255).astype(np.uint8)
img_blur = cv2.blur(img, (kernel_size, kernel_size))
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('原始图像')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(g, cmap='gray')
plt.title(f'频域均值滤波 ({kernel_size}x{kernel_size})')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(img_blur, cmap='gray')
plt.title('空间域 cv2.blur（参考）')
plt.axis('off')
plt.tight_layout()
plt.show()