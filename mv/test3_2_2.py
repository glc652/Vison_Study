"""
2-2.jpg 专用形状检测
检测 4 个物体：
- 直尺 -> 矩形 (蓝色)
- 塑料盒 -> 矩形 (绿色)
- 三角尺 -> 三角形 (蓝色)
- 胶带 -> 圆形 (红色)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def detect_shapes_2_2(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shapes = []

    # 自适应阈值检测轮廓
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        rect = cv2.minAreaRect(contour)
        (cx, cy), (bw, bh), _ = rect
        rect_area = bw * bh
        rect_fill_ratio = area / rect_area if rect_area > 0 else 0
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 凸包顶点数（用较大epsilon，更稳定）
        hull = cv2.convexHull(contour)
        hull_approx = cv2.approxPolyDP(hull, 0.04 * perimeter, True)

        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = int(cx), int(cy)

        candidates.append({
            'contour': contour,
            'area': area,
            'circularity': circularity,
            'vertices': len(approx),
            'hull_vertices': len(hull_approx),
            'aspect_ratio': aspect_ratio,
            'rect_fill_ratio': rect_fill_ratio,
            'center': (cx, cy),
            'approx': approx
        })

    # 按面积排序，取前 4 个
    candidates.sort(key=lambda x: x['area'], reverse=True)
    selected = candidates[:4]

    for c in selected:
        name = None
        shape_type = None

        # 胶带：圆形度高
        if c['circularity'] > 0.6:
            name = '胶带'
            shape_type = 'Circle'
        # 直尺：长宽比大
        elif c['aspect_ratio'] > 3.0:
            name = '直尺'
            shape_type = 'Rectangle'
        # 三角尺：凸包用较大epsilon拟合后为3个顶点
        elif c['hull_vertices'] == 3:
            name = '三角尺'
            shape_type = 'Triangle'
        # 塑料盒：其余矩形
        else:
            name = '塑料盒'
            shape_type = 'Rectangle'

        entry = {
            'name': name,
            'type': shape_type,
            'center': c['center'],
            'contour': c['contour'],
            'vertices': c['vertices'],
            'aspect_ratio': c['aspect_ratio'],
            'rect_fill_ratio': c['rect_fill_ratio']
        }
        if shape_type == 'Circle':
            _, radius = cv2.minEnclosingCircle(c['contour'])
            entry['radius'] = int(radius)
        shapes.append(entry)

    return original, adaptive, shapes


def put_chinese_text(img, text, pos, color, font_size=20):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_results(img, shapes):
    result = img.copy()

    colors = {
        '三角尺': (0, 255, 0),
        '直尺':   (255, 0, 0),
        '塑料盒': (255, 0, 0),
        '胶带':   (0, 0, 255)
    }

    for shape in shapes:
        name = shape['name']
        center = shape['center']
        cx, cy = center
        color = colors.get(name, (0, 255, 255))

        if shape['type'] == 'Circle':
            radius = shape.get('radius', 50)
            cv2.circle(result, center, radius, color, 3)
            cv2.circle(result, center, 5, (0, 255, 0), -1)
        elif shape['type'] == 'Triangle':
            contour = shape['contour']
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) != 3:
                hull = cv2.convexHull(contour)
                epsilon = 0.04 * perimeter
                approx = cv2.approxPolyDP(hull, epsilon, True)
            cv2.drawContours(result, [approx], -1, color, 3)
            cv2.circle(result, center, 5, (0, 255, 0), -1)
        elif shape['type'] == 'Rectangle':
            contour = shape['contour']
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(result, [box], 0, color, 3)
            cv2.circle(result, center, 5, (0, 255, 0), -1)

        result = put_chinese_text(result, name, (cx - 30, cy - 30), color)

    return result


def print_positions(shapes):
    print("\n" + "=" * 60)
    print("2-2.jpg 形状检测结果")
    print("=" * 60)

    for shape in shapes:
        name = shape['name']
        cx, cy = shape['center']
        if shape['type'] == 'Circle':
            print(f"[{name}] 中心：({cx}, {cy}), 半径：{shape.get('radius', 'N/A')}")
        else:
            print(f"[{name}] 中心：({cx}, {cy}), 面积：{cv2.contourArea(shape.get('contour', [])):.0f}")

    print(f"\n共检测到 {len(shapes)} 个物体")


if __name__ == "__main__":
    image_path = r"D:\vision\mv\2-2.jpg"

    print(f"处理：{image_path}")

    img, thresh, shapes = detect_shapes_2_2(image_path)
    print_positions(shapes)

    result = draw_results(img, shapes)

    fig, axes = plt.subplots(1, 3, figsize=(16, 12))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('2-2 - 原图')
    axes[0].axis('off')

    axes[1].imshow(thresh, cmap='gray')
    axes[1].set_title('自适应阈值')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title('检测结果')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite(r"D:\vision\mv\2-2_result.jpg", result)
    cv2.imwrite(r"D:\vision\mv\2-2_thresh.jpg", thresh)
    print("\n结果已保存到：D:\\vision\\mv\\2-2_result.jpg")
