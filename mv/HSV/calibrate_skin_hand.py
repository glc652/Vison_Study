import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def nothing(x):
    pass

fish_dir = r"D:\vision\mv\HSV\fish"
images = sorted(Path(fish_dir).glob("*.jpg"))

img_5 = cv2.imdecode(np.fromfile(str(images[4]), dtype=np.uint8), cv2.IMREAD_COLOR)

cv2.namedWindow('Skin Detection Calibration')

cv2.createTrackbar('H_Lower', 'Skin Detection Calibration', 0, 180, nothing)
cv2.createTrackbar('H_Upper', 'Skin Detection Calibration', 20, 180, nothing)
cv2.createTrackbar('S_Lower', 'Skin Detection Calibration', 20, 255, nothing)
cv2.createTrackbar('S_Upper', 'Skin Detection Calibration', 255, 255, nothing)
cv2.createTrackbar('V_Lower', 'Skin Detection Calibration', 100, 255, nothing)
cv2.createTrackbar('V_Upper', 'Skin Detection Calibration', 255, 255, nothing)

hsv_5 = cv2.cvtColor(img_5, cv2.COLOR_BGR2HSV)

print("调整滑块找到最佳肤色范围")
print("按 's' 保存参数")
print("按 'q' 退出")

cv2.imshow('Skin Detection Calibration', np.zeros((100, 400), dtype=np.uint8))

plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

while True:
    h_lower = cv2.getTrackbarPos('H_Lower', 'Skin Detection Calibration')
    h_upper = cv2.getTrackbarPos('H_Upper', 'Skin Detection Calibration')
    s_lower = cv2.getTrackbarPos('S_Lower', 'Skin Detection Calibration')
    s_upper = cv2.getTrackbarPos('S_Upper', 'Skin Detection Calibration')
    v_lower = cv2.getTrackbarPos('V_Lower', 'Skin Detection Calibration')
    v_upper = cv2.getTrackbarPos('V_Upper', 'Skin Detection Calibration')

    lower = np.array([h_lower, s_lower, v_lower])
    upper = np.array([h_upper, s_upper, v_upper])

    mask_5 = cv2.inRange(hsv_5, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_5 = cv2.morphologyEx(mask_5, cv2.MORPH_CLOSE, kernel)
    mask_5 = cv2.morphologyEx(mask_5, cv2.MORPH_OPEN, kernel)

    result_5 = cv2.bitwise_and(img_5, img_5, mask=mask_5)

    text = f"H: {h_lower}-{h_upper} | S: {s_lower}-{s_upper} | V: {v_lower}-{v_upper}"
    fig.suptitle(text, fontsize=14)

    img_5_rgb = cv2.cvtColor(img_5, cv2.COLOR_BGR2RGB)
    result_5_rgb = cv2.cvtColor(result_5, cv2.COLOR_BGR2RGB)

    axes[0].clear()
    axes[0].imshow(img_5_rgb)
    axes[0].set_title("Image 5 - Original")
    axes[0].axis('off')

    axes[1].clear()
    axes[1].imshow(result_5_rgb)
    axes[1].set_title("Image 5 - Skin Detection")
    axes[1].axis('off')

    axes[2].clear()
    axes[2].imshow(mask_5, cmap='gray')
    axes[2].set_title("Image 5 - Mask")
    axes[2].axis('off')

    plt.tight_layout()
    plt.pause(0.01)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print(f"\n保存的肤色 HSV 参数:")
        print(f"lower_skin = np.array([{h_lower}, {s_lower}, {v_lower}])")
        print(f"upper_skin = np.array([{h_upper}, {s_upper}, {v_upper}])")
        print("\n将这些值复制到 detect_fish.py 中")

cv2.destroyAllWindows()
plt.close()
