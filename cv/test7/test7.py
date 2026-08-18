import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

image_path = r"D:\vision\cv\images\Lena.tif"
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

def add_salt_pepper_noise(image, prob=0.2):
    noisy = image.copy()
    num_noisy = int(prob * image.size)
    
    coords = np.random.choice(image.size, num_noisy, replace=False)
    coords = np.unravel_index(coords, image.shape)
    
    half = num_noisy // 2
    noisy[coords[0][:half], coords[1][:half]] = 0      
    noisy[coords[0][half:], coords[1][half:]] = 255    
    return noisy

noisy_img = add_salt_pepper_noise(img, prob=0.2)
filtered_3x3 = cv.medianBlur(noisy_img, ksize=3)
filtered_5x5 = cv.medianBlur(noisy_img, ksize=5)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
images = [img, noisy_img, filtered_3x3, filtered_5x5]
titles = ['原始图像', '椒盐噪声（密度0.2）', '3×3中值滤波', '5×5中值滤波']
for ax, im, title in zip(axes, images, titles):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()