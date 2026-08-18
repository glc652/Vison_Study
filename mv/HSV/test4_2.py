import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread(r"D:\vision\cv\images\Fig0301.png")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 分离通道 (OpenCV: B, G, R)
B, G, R = cv2.split(img_bgr)

# 构造三幅单色图像（保持三通道格式，以便转 HSV）
red_only   = np.zeros_like(img_bgr)
red_only[:, :, 2] = R  # OpenCV 中 R 在第2通道

green_only = np.zeros_like(img_bgr)
green_only[:, :, 1] = G

blue_only  = np.zeros_like(img_bgr)
blue_only[:, :, 0] = B

single_color_images = {
    'Red Channel': red_only,
    'Green Channel': green_only,
    'Blue Channel': blue_only
}

# 设置绘图
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle('RGB and HSV ', fontsize=14)

for i, (name, img_bgr_single) in enumerate(single_color_images.items()):
    # 转换为 HSV
    img_hsv = cv2.cvtColor(img_bgr_single, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    
    # 计算平均 HSV（忽略全黑区域可选，此处简单平均）
    h_mean = H.mean()
    s_mean = S.mean()
    v_mean = V.mean()
    print(f"{name} → Avg HSV: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
    
    # 显示原单色图（转为 RGB 显示）
    img_rgb_display = cv2.cvtColor(img_bgr_single, cv2.COLOR_BGR2RGB)
    axes[i, 0].imshow(img_rgb_display)
    axes[i, 0].set_title(f'{name}')
    axes[i, 0].axis('off')
    
    # 显示 H, S, V 通道
    axes[i, 1].imshow(H, cmap='hsv')
    axes[i, 1].set_title(f'H (avg={h_mean:.0f})')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(S, cmap='gray')
    axes[i, 2].set_title(f'S (avg={s_mean:.0f})')
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(V, cmap='gray')
    axes[i, 3].set_title(f'V (avg={v_mean:.0f})')
    axes[i, 3].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()