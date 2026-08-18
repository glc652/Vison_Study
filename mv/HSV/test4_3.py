import cv2
import numpy as np
import matplotlib.pyplot as plt

def skin_detection_hsv(img):
    """肤色检测"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([10, 40, 60])
    upper = np.array([29, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # 膨胀+腐蚀，让掩码更平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def face_beautify(img, beauty_level, white_level):
    """
    人脸美颜（只美颜皮肤区域）
    :param img: 原图
    :param beauty_level: 磨皮强度 1~10
    :param white_level: 美白强度 1~20
    :return: 美颜后图像
    """
    # 1. 获取皮肤掩码
    skin_mask = skin_detection_hsv(img)
    
    # 2. 磨皮（双边滤波，保边磨皮）
    d = 8 + beauty_level  # 滤波直径
    sigma = 50 + beauty_level * 5
    blur = cv2.bilateralFilter(img, d, sigma, sigma)

    # 3. 美白（提亮皮肤）
    M = np.ones(img.shape, dtype=np.float32)
    # white_img = img.astype(np.float32) + white_level * 2.55#cv2.addWeighted(img, 1, M, 0.01 * white_level, 0)
    # white_img = np.clip(white_img, 0, 255).astype(np.uint8)
    blur_white = blur.astype(np.float32) + white_level * 2.55
    blur_white = np.clip(blur_white, 0, 255).astype(np.uint8)

    # 4. 融合：只把美颜效果应用到皮肤区域
    result = img.copy()
    # result[skin_mask > 0] = blur[skin_mask > 0]  # 磨皮
    result[skin_mask > 0] = blur_white[skin_mask > 0]  # 美白
    
    return result

# ==================== 测试 ====================
if __name__ == "__main__":
    img = cv2.imread(r"D:\vision\cv\images\Fig0903.png")  
    output = face_beautify(img, 10, 1)

    img_show = cv2.resize(img, (640, 800))
    img_show2 = cv2.resize(output, (640, 800))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # cv2.imshow("Original", img_show)
    # cv2.imshow("Beautify", img_show2)
    plt.figure(figsize=(12, 6))  # 宽12英寸，高6英寸
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis('off')  # 关闭坐标轴
    
    plt.subplot(1, 2, 2)
    plt.imshow(output_rgb)
    plt.title("Beautify")
    plt.axis('off')
    
    plt.tight_layout()

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()