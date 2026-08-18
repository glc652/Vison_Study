# # import cv2
# # import matplotlib.pyplot as plt

# # # 读取图像
# # logo = cv2.imread("D:\\vision\\cv\\images\\LogoCV.png")
# # bg = cv2.imread("D:\\vision\\cv\\images\\Frowns.jpg")

# # if logo is None or bg is None:
# #     print("请确保 logo.png 和 background.jpg 存在")
# #     exit()

# # # 调整背景图大小以匹配logo
# # bg = cv2.resize(bg, (logo.shape[1], logo.shape[0]))

# # # 直接替换 
# # method1 = bg.copy()
# # h, w = logo.shape[:2]
# # method1[:h, :w] = logo

# # # 普通加法 
# # method2 = cv2.addWeighted(bg, 0.7, logo, 0.3, 0)

# # # 掩码透明叠加
# # # 1. 灰度化
# # gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
# # # 2. 二值化生成掩码
# # _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# # # 3. 反转掩码（背景区域为白色，前景为黑色）
# # mask_inv = cv2.bitwise_not(mask)
# # # 4. 提取前景（logo中非透明区域）
# # fg = cv2.bitwise_and(logo, logo, mask=mask)
# # # 5. 提取背景中要放置logo的区域
# # roi = cv2.bitwise_and(bg, bg, mask=mask_inv)
# # # 6. 合并前景和背景
# # method3 = cv2.add(roi, fg)


# # plt.figure(figsize=(12, 8))
# # plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)), plt.title('Background')
# # plt.axis('off')
# # plt.subplot(2, 3, 2), plt.imshow(cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)), plt.title('Logo')
# # plt.axis('off')
# # plt.subplot(2, 3, 3), plt.imshow(mask, cmap='gray'), plt.title('Mask')
# # plt.axis('off')
# # plt.subplot(2, 3, 4), plt.imshow(cv2.cvtColor(method1, cv2.COLOR_BGR2RGB)), plt.title('Method 1: Direct Replace')
# # plt.axis('off')
# # plt.subplot(2, 3, 5), plt.imshow(cv2.cvtColor(method2, cv2.COLOR_BGR2RGB)), plt.title('Method 2: Weighted Blend')
# # plt.axis('off')
# # plt.subplot(2, 3, 6), plt.imshow(cv2.cvtColor(method3, cv2.COLOR_BGR2RGB)), plt.title('Method 3: Mask-based Overlay')
# # plt.axis('off')

# # plt.tight_layout()
# # plt.savefig('result.png', dpi=150)
# # plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.resize(img, (400, 300))
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# value = 100

# # ========== 1. 彩色图像 + 常数（标量）==========
# # OpenCV 饱和加法（自动限制在 [0, 255]）
# opencv_add_color = cv2.add(img, np.array([value, value, value], dtype=np.uint8))

# # NumPy 模运算（溢出后回绕，如 250+10=4）
# numpy_add_color = img + value  # 自动 uint8 模 256

# # ========== 2. 灰度图像 + 常数 ==========
# opencv_add_gray = cv2.add(gray_img, value)
# numpy_add_gray = gray_img + value

# # ========== 显示结果 ==========
# plt.figure(figsize=(12, 10))

# plt.subplot(3, 3, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original (Color)'), plt.axis('off')
# plt.subplot(3, 3, 2), plt.imshow(gray_img, cmap='gray'), plt.title('Original (Gray)'), plt.axis('off')
# # OpenCV 加法
# plt.subplot(3, 3, 4), plt.imshow(cv2.cvtColor(opencv_add_color, cv2.COLOR_BGR2RGB)), plt.title('OpenCV Add (Color)'), plt.axis('off')
# plt.subplot(3, 3, 5), plt.imshow(opencv_add_gray, cmap='gray'), plt.title('OpenCV Add (Gray)'), plt.axis('off')
# # NumPy 加法
# plt.subplot(3, 3, 7), plt.imshow(cv2.cvtColor(numpy_add_color, cv2.COLOR_BGR2RGB)), plt.title('NumPy Add (Color)'), plt.axis('off')
# plt.subplot(3, 3, 8), plt.imshow(numpy_add_gray, cmap='gray'), plt.title('NumPy Add (Gray)'), plt.axis('off')
# # 差异对比
# diff_color = np.abs(opencv_add_color.astype(int) - numpy_add_color.astype(int)).astype(np.uint8)
# diff_gray = np.abs(opencv_add_gray.astype(int) - numpy_add_gray.astype(int)).astype(np.uint8)
# plt.subplot(3, 3, 6), plt.imshow(diff_color.mean(axis=2), cmap='hot'), plt.title('Diff: Color'), plt.axis('off')
# plt.subplot(3, 3, 9), plt.imshow(diff_gray, cmap='hot'), plt.title('Diff: Gray'), plt.axis('off')
# plt.tight_layout()
# plt.savefig('addition_comparison.png', dpi=150)
# plt.show()

import cv2
import numpy as np

def adjust_brightness(img, beta):
    result = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
    return result

def adjust_contrast(img, alpha):
    result = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return result

def adjust_brightness_contrast(img, alpha, beta):
    result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return result

def adjust_hue(img, hue_shift):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h = h.astype(np.int16)
    h = (h + hue_shift) % 180
    h = h.astype(np.uint8)

    hsv_result = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)
    return result

def print_image_info(name, img):
    mean_val = cv2.mean(img)[:3]
    print(f"{name}:")
    print(f"  尺寸 = {img.shape}")
    print(f"  BGR三通道均值 = {mean_val}")
    print("-" * 40)

def add_title(img, title):
    """
    在图片上方添加标题
    """
    h, w = img.shape[:2]
    title_area = np.ones((40, w, 3), dtype=np.uint8) * 255  # 白色标题栏

    cv2.putText(
        title_area,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    result = np.vstack((title_area, img))
    return result

def make_collage(images, titles, target_size=(300, 220)):
    """
    images: 图像列表
    titles: 标题列表
    target_size: 每张小图统一尺寸 (宽, 高)
    """
    processed = []

    for img, title in zip(images, titles):
        resized = cv2.resize(img, target_size)
        titled = add_title(resized, title)
        processed.append(titled)

    row1 = np.hstack(processed[:4])
    row2 = np.hstack(processed[4:8])
    collage = np.vstack((row1, row2))
    return collage

def main():
    img = cv2.imread("test.jpg")

    if img is None:
        print("错误：图像读取失败，请检查文件路径是否正确！")
        return

    brighter = adjust_brightness(img, 50)
    darker = adjust_brightness(img, -50)
    high_contrast = adjust_contrast(img, 1.5)
    low_contrast = adjust_contrast(img, 0.6)
    bc_result = adjust_brightness_contrast(img, 1.2, 30)
    hue_plus = adjust_hue(img, 30)
    hue_minus = adjust_hue(img, -30)
    print_image_info("原图", img)
    print_image_info("亮度增加", brighter)
    print_image_info("亮度降低", darker)
    print_image_info("对比度增强", high_contrast)
    print_image_info("对比度降低", low_contrast)
    print_image_info("亮度+对比度调整", bc_result)
    print_image_info("色调增加", hue_plus)
    print_image_info("色调降低", hue_minus)
    images = [
        img, brighter, darker, high_contrast,
        low_contrast, bc_result, hue_plus, hue_minus
    ]

    titles = [
        "Original", "Brighter", "Darker", "High Contrast",
        "Low Contrast", "Brightness+Contrast", "Hue +30", "Hue -30"
    ]

    collage = make_collage(images, titles, target_size=(300, 220))
    cv2.imwrite("result_collage.jpg", collage)
    cv2.imshow("All Results (2x4)", collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()