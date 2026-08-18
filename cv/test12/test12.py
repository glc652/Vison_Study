import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path = r"D:\vision\cv\images\Fig1602.png"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
original = img_rgb.copy()
def adjust_saturation_rgb_space(img_rgb, factor):
    """在RGB空间中调整饱和度（实际通过HSV转换实现）"""
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

rgb_increased = adjust_saturation_rgb_space(img_rgb, 1.5)   # 增大饱和度
rgb_decreased = adjust_saturation_rgb_space(img_rgb, 0.5)   # 减小饱和度

def adjust_saturation_hsv_space(img_rgb, factor):
    """在HSV空间中直接调整饱和度"""
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

hsv_increased = adjust_saturation_hsv_space(img_rgb, 1.5)   # 增大饱和度
hsv_decreased = adjust_saturation_hsv_space(img_rgb, 0.5)   # 减小饱和度

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(original)
plt.title('Original Image')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(rgb_increased)
plt.title('RGB Space - Increased Saturation')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(rgb_decreased)
plt.title('RGB Space - Decreased Saturation')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(hsv_increased)
plt.title('HSV Space - Increased Saturation')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(hsv_decreased)
plt.title('HSV Space - Decreased Saturation')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.axis('off')
plt.tight_layout()
plt.show()