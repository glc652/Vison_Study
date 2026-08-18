"""
2-1.jpg 专用形状检测
检测 4 个物体：
- 直尺 -> 矩形 (蓝色)
- 塑料盒 -> 矩形 (绿色)
- 三角尺 -> 三角形 (蓝色)
- 胶带 -> 圆形 (红色)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def detect_shapes_2_1(image_path):
    """
    针对 2-1.jpg 的形状检测
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shapes = []

    # ========== 1. 检测胶带（黑色圆形） ==========
    _, tape_thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel_tape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tape_closed = cv2.morphologyEx(tape_thresh, cv2.MORPH_CLOSE, kernel_tape, iterations=5)
    tape_contours, _ = cv2.findContours(tape_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tape_center = None
    tape_radius = 0

    for contour in tape_contours:
        area = cv2.contourArea(contour)
        if 50000 < area < 800000:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:
                    (cx, cy), radius = cv2.minEnclosingCircle(contour)
                    tape_center = (int(cx), int(cy))
                    tape_radius = int(radius)
                    shapes.append({
                        'name': '胶带',
                        'type': 'Circle',
                        'center': tape_center,
                        'radius': tape_radius,
                        'contour': contour
                    })
                    break

    # ========== 2. 检测透明物体 ==========
    # CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # 高斯模糊去除刻度噪声
    blur = cv2.GaussianBlur(enhanced, (11, 11), 0)

    # Canny 边缘检测
    edges = cv2.Canny(blur, 50, 150)

    # 形态学处理 - 连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)
    dilated = cv2.dilate(closed, kernel, iterations=5)

    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 存储候选轮廓
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # 面积阈值：只检测大物体
        if area < 300000:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        hull = cv2.convexHull(contour)
        hull_approx = hull
        for eps_f in [0.04, 0.05, 0.06, 0.08, 0.10]:
            hull_approx = cv2.approxPolyDP(hull, eps_f * perimeter, True)
            if len(hull_approx) == 3:
                break

        rect = cv2.minAreaRect(contour)
        (cx, cy), (bw, bh), angle = rect
        rect_area = bw * bh
        rect_fill_ratio = area / rect_area if rect_area > 0 else 0
        aspect_ratio = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 1

        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = int(cx), int(cy)

        # 跳过与胶带太近的轮廓
        if tape_center:
            dist = np.sqrt((cx - tape_center[0])**2 + (cy - tape_center[1])**2)
            if dist < 200:
                continue

        candidates.append({
            'contour': contour,
            'area': area,
            'vertices': vertices,
            'hull_vertices': len(hull_approx),
            'aspect_ratio': aspect_ratio,
            'rect_fill_ratio': rect_fill_ratio,
            'center': (cx, cy),
            'approx': approx
        })

    # 按面积排序，取前 3 个最大的（直尺、三角尺、塑料盒）
    candidates.sort(key=lambda x: x['area'], reverse=True)
    selected = candidates[:3] if len(candidates) > 3 else candidates

    # 形状分类
    for c in selected:
        name = None
        shape_type = None

        # 直尺：长宽比大
        if c['aspect_ratio'] > 2.5:
            name = '直尺'
            shape_type = 'Rectangle'
        # 三角尺：凸包拟合为3个顶点，且矩形填充率低（三角形约0.5）
        elif c['hull_vertices'] == 3 and c['rect_fill_ratio'] < 0.65:
            name = '三角尺'
            shape_type = 'Triangle'
        # 塑料盒：矩形填充率高的大轮廓
        else:
            name = '塑料盒'
            shape_type = 'Rectangle'

        shapes.append({
            'name': name,
            'type': shape_type,
            'center': c['center'],
            'contour': c['contour'],
            'vertices': c['vertices'],
            'aspect_ratio': c['aspect_ratio'],
            'rect_fill_ratio': c['rect_fill_ratio']
        })

    return original, edges, shapes


def draw_results(img, shapes):
    """绘制结果"""
    result = img.copy()

    colors = {
        '三角尺': (0, 255, 0),      # 绿色
        '直尺': (255, 0, 0),        # 蓝色
        '塑料盒': (255, 0, 0),      # 绿色
        '胶带': (0, 0, 255)         # 红色
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
            # 画三角形（使用多边形拟合）
            contour = shape['contour']
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # 使用凸包拟合三角形，逐步增大epsilon直到得到3个顶点
            hull = cv2.convexHull(contour)
            for eps_factor in [0.04, 0.05, 0.06, 0.08, 0.10]:
                approx = cv2.approxPolyDP(hull, eps_factor * perimeter, True)
                if len(approx) == 3:
                    break
            cv2.drawContours(result, [approx], -1, color, 3)
            cv2.circle(result, center, 5, (0, 255, 0), -1)
        elif shape['type'] == 'Rectangle':
            # 画矩形（最小外接矩形）
            contour = shape['contour']
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(result, [box], 0, color, 3)
            cv2.circle(result, center, 5, (0, 255, 0), -1)

        cv2.putText(result, name, (cx - 30, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result


def print_positions(shapes):
    """打印位置信息"""
    print("\n" + "=" * 60)
    print("2-1.jpg 形状检测结果")
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
    image_path = r"D:\vision\mv\2-1_original.jpg"

    print(f"处理：{image_path}")

    img, edges, shapes = detect_shapes_2_1(image_path)
    print_positions(shapes)

    result = draw_results(img, shapes)

    fig, axes = plt.subplots(1, 3, figsize=(16, 12))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('2-1 - 原图')
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('边缘检测')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title('检测结果')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite(r"D:\vision\mv\2-3_test_result.jpg", result)
    cv2.imwrite(r"D:\vision\mv\2-3_test_edges.jpg", edges)
    print("\n结果已保存到：D:\\vision\\mv\\2-3_test_result.jpg")
