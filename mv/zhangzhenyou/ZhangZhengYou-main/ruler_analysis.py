import os
import cv2
import numpy as np
from test import load_calibration_result


def calculate_pixel_to_mm_ratio(img, square_size_mm=15, points_per_row=11, points_per_col=8):
    """从棋盘格计算像素到毫米的转换比例"""
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (points_per_row, points_per_col), None)

    if not ret:
        print("未检测到棋盘格，无法计算转换比例")
        return None

    # 精化角点位置
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 计算棋盘格在图像中的像素尺寸
    # 棋盘格有 (points_per_row-1) x (points_per_col-1) 个方格
    corners_reshaped = corners_refined.reshape((points_per_col, points_per_row, 2))

    # 计算水平方向的平均像素距离（相邻两个角点之间）
    horizontal_distances = []
    for row in corners_reshaped:
        for i in range(len(row) - 1):
            dist = np.linalg.norm(row[i+1] - row[i])
            horizontal_distances.append(dist)
    avg_horizontal_px = np.mean(horizontal_distances)

    # 计算竖直方向的平均像素距离
    vertical_distances = []
    for col_idx in range(points_per_row):
        for row_idx in range(points_per_col - 1):
            dist = np.linalg.norm(corners_reshaped[row_idx+1, col_idx] - corners_reshaped[row_idx, col_idx])
            vertical_distances.append(dist)
    avg_vertical_px = np.mean(vertical_distances)

    # 平均像素距离（一个方格的边长）
    avg_px_per_square = (avg_horizontal_px + avg_vertical_px) / 2

    # 转换比例：mm/pixel
    ratio_mm_per_px = square_size_mm / avg_px_per_square

    print(f"棋盘格检测成功:")
    print(f"  水平方向平均像素距离: {avg_horizontal_px:.2f} px")
    print(f"  竖直方向平均像素距离: {avg_vertical_px:.2f} px")
    print(f"  平均每个方格: {avg_px_per_square:.2f} px")
    print(f"  转换比例: {ratio_mm_per_px:.4f} mm/px\n")

    return ratio_mm_per_px


def remove_shadow_edges(img, margin_percent=5):
    """移除图像边缘的阴影部分"""
    h, w = img.shape[:2]
    margin_h = int(h * margin_percent / 100)
    margin_w = int(w * margin_percent / 100)

    # 裁剪边缘
    img_cropped = img[margin_h:h-margin_h, margin_w:w-margin_w]

    return img_cropped


def undistort_image(img, A, k):
    """对图像进行畸变校正，避免黑边"""
    h, w = img.shape[:2]
    k_full = np.concatenate([k, np.zeros(3)])

    # 获取最优的新相机矩阵
    # alpha=1: 保留所有像素但可能有黑边，alpha=0: 完全裁剪黑边
    new_A, roi = cv2.getOptimalNewCameraMatrix(A, k_full, (w, h), alpha=1)

    # 使用新的相机矩阵进行校正
    map_x, map_y = cv2.initUndistortRectifyMap(A, k_full, None, new_A, (w, h), 5)
    img_undistorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    # 手动移除黑边
    img_undistorted = remove_black_borders(img_undistorted, threshold=5)

    return img_undistorted


def remove_black_borders(img, threshold=5):
    """移除图像周围的黑边"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 找到非黑色像素的边界
    rows = np.where(np.max(gray, axis=1) > threshold)[0]
    cols = np.where(np.max(gray, axis=0) > threshold)[0]

    if len(rows) == 0 or len(cols) == 0:
        return img

    y_min, y_max = rows[0], rows[-1]
    x_min, x_max = cols[0], cols[-1]

    # 裁剪图像
    img_cropped = img[y_min:y_max+1, x_min:x_max+1]

    return img_cropped


def detect_rulers_with_edges(img):
    """使用边缘检测和Hough变换检测尺子"""
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 创建掩码来屏蔽黑边
    # 找到非黑色像素的边界
    rows = np.where(np.max(gray, axis=1) > 20)[0]
    cols = np.where(np.max(gray, axis=0) > 20)[0]

    if len(rows) > 0 and len(cols) > 0:
        y_min, y_max = rows[0], rows[-1]
        x_min, x_max = cols[0], cols[-1]

        # 创建掩码
        mask = np.zeros_like(gray)
        mask[y_min:y_max+1, x_min:x_max+1] = 255

        # 应用掩码
        gray_masked = cv2.bitwise_and(gray, mask)
    else:
        gray_masked = gray

    # 应用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 膨胀边缘以连接断裂的线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # 使用Hough线变换检测直线 - 多次尝试不同参数
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=30, maxLineGap=30)

    return edges, lines


def detect_corners_with_harris(edges, corner_pos, window_size=25, use_subpixel=True):
    """在边缘图像上使用Harris角点检测精化角点位置，支持亚像素精度"""
    h, w = edges.shape[:2]
    x, y = int(corner_pos[0]), int(corner_pos[1])

    # 定义搜索窗口
    half_win = window_size // 2
    x_min = max(0, x - half_win)
    x_max = min(w, x + half_win + 1)
    y_min = max(0, y - half_win)
    y_max = min(h, y + half_win + 1)

    window = edges[y_min:y_max, x_min:x_max].astype(np.float32)

    if window.size < 9 or np.sum(window) < 10:
        return corner_pos

    # Harris角点检测
    harris_response = cv2.cornerHarris(window, blockSize=3, ksize=3, k=0.04)

    # 找到最强响应的位置
    if harris_response.max() > 0:
        y_local, x_local = np.unravel_index(np.argmax(harris_response), harris_response.shape)

        # 亚像素精化：使用周围像素的加权平均
        if use_subpixel and y_local > 0 and y_local < harris_response.shape[0]-1 and \
           x_local > 0 and x_local < harris_response.shape[1]-1:
            # 获取3x3邻域
            neighborhood = harris_response[y_local-1:y_local+2, x_local-1:x_local+2]
            # 计算加权中心
            weights = neighborhood / (neighborhood.sum() + 1e-10)
            y_offset = np.sum(weights * np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
            x_offset = np.sum(weights * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))

            refined_x = x_min + x_local + x_offset
            refined_y = y_min + y_local + y_offset
        else:
            refined_x = x_min + x_local
            refined_y = y_min + y_local

        return np.array([refined_x, refined_y], dtype=np.float32)

    return corner_pos


def find_edge_boundary(edges, corner_pos, direction='right', search_range=50):
    """在边缘图像上沿指定方向找到最近的边缘边界"""
    h, w = edges.shape[:2]
    x, y = int(corner_pos[0]), int(corner_pos[1])

    if direction == 'right':
        # 从当前位置向右搜索，找到最右边的边缘像素
        for dx in range(search_range):
            check_x = x + dx
            if check_x >= w:
                break
            # 在y方向上检查一个小范围
            for dy in range(-5, 6):
                check_y = y + dy
                if 0 <= check_y < h and edges[check_y, check_x] > 0:
                    return np.array([check_x, check_y], dtype=np.float32)
        return corner_pos

    elif direction == 'left':
        # 从当前位置向左搜索
        for dx in range(search_range):
            check_x = x - dx
            if check_x < 0:
                break
            for dy in range(-5, 6):
                check_y = y + dy
                if 0 <= check_y < h and edges[check_y, check_x] > 0:
                    return np.array([check_x, check_y], dtype=np.float32)
        return corner_pos

    elif direction == 'top':
        # 从当前位置向上搜索
        for dy in range(search_range):
            check_y = y - dy
            if check_y < 0:
                break
            for dx in range(-5, 6):
                check_x = x + dx
                if 0 <= check_x < w and edges[check_y, check_x] > 0:
                    return np.array([check_x, check_y], dtype=np.float32)
        return corner_pos

    elif direction == 'bottom':
        # 从当前位置向下搜索
        for dy in range(search_range):
            check_y = y + dy
            if check_y >= h:
                break
            for dx in range(-5, 6):
                check_x = x + dx
                if 0 <= check_x < w and edges[check_y, check_x] > 0:
                    return np.array([check_x, check_y], dtype=np.float32)
        return corner_pos

    return corner_pos


def refine_corner_with_gradient(img, corner_pos, window_size=15):
    """使用梯度信息精化角点位置到亚像素精度"""
    h, w = img.shape[:2]
    x, y = int(corner_pos[0]), int(corner_pos[1])

    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 定义搜索窗口
    half_win = window_size // 2
    x_min = max(0, x - half_win)
    x_max = min(w, x + half_win + 1)
    y_min = max(0, y - half_win)
    y_max = min(h, y + half_win + 1)

    window = gray[y_min:y_max, x_min:x_max].astype(np.float32)

    if window.size < 9:
        return corner_pos

    # 计算梯度
    gx = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)

    # 计算Harris角点响应
    gx2 = gx * gx
    gy2 = gy * gy

    # 应用高斯加权
    kernel = cv2.getGaussianKernel(window_size, window_size / 4)
    kernel_2d = kernel @ kernel.T

    # 找到最强响应的位置
    response = gx2 * gy2 - 0.04 * (gx2 + gy2) ** 2
    response_weighted = response * kernel_2d[:response.shape[0], :response.shape[1]]

    if response_weighted.size == 0:
        return corner_pos

    # 找到最大响应位置
    y_local, x_local = np.unravel_index(np.argmax(response_weighted), response_weighted.shape)

    # 转换回原始图像坐标
    refined_x = x_min + x_local
    refined_y = y_min + y_local

    return np.array([refined_x, refined_y], dtype=np.float32)


def find_corners_from_coords(x_coords, y_coords, img=None, edges=None, use_harris=True):
    """从坐标集合中找到四个角点，可选使用Harris角点检测"""
    if len(x_coords) < 4 or len(y_coords) < 4:
        return None

    # 创建所有可能的角点候选
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # 找到最接近四个角的点
    points = np.column_stack([x_coords, y_coords])

    # 左上角 (x_min, y_min)
    top_left_idx = np.argmin((points[:, 0] - x_min)**2 + (points[:, 1] - y_min)**2)
    top_left = points[top_left_idx]

    # 右上角 (x_max, y_min)
    top_right_idx = np.argmin((points[:, 0] - x_max)**2 + (points[:, 1] - y_min)**2)
    top_right = points[top_right_idx]

    # 右下角 (x_max, y_max)
    bottom_right_idx = np.argmin((points[:, 0] - x_max)**2 + (points[:, 1] - y_max)**2)
    bottom_right = points[bottom_right_idx]

    # 左下角 (x_min, y_max)
    bottom_left_idx = np.argmin((points[:, 0] - x_min)**2 + (points[:, 1] - y_max)**2)
    bottom_left = points[bottom_left_idx]

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    # 使用Harris角点检测精化角点
    if use_harris and edges is not None:
        refined_corners = []
        for i, corner in enumerate(corners):
            # 左端角点使用更大的搜索窗口（因为黑影影响）
            if i in [0, 3]:  # 左上角和左下角
                window_size = 100
            # 右端角点使用较大的搜索窗口
            elif i in [1, 2]:  # 右上角和右下角
                window_size = 35
            else:
                window_size = 25
            refined = detect_corners_with_harris(edges, corner, window_size=window_size)

            # 对各端角点进行边界精化
            if i in [1, 2]:  # 右上角和右下角
                refined = find_edge_boundary(edges, refined, direction='right', search_range=40)
            elif i in [0, 3]:  # 左上角和左下角
                refined = find_edge_boundary(edges, refined, direction='left', search_range=30)

            # 对上下端角点进行额外的边界精化
            if i in [0, 1]:  # 上端角点
                refined = find_edge_boundary(edges, refined, direction='top', search_range=30)
            elif i in [2, 3]:  # 下端角点
                refined = find_edge_boundary(edges, refined, direction='bottom', search_range=15)

            refined_corners.append(refined)
        corners = np.array(refined_corners, dtype=np.int32)
    elif img is not None:
        # 备选：使用梯度精化
        refined_corners = []
        for corner in corners:
            refined = refine_corner_with_gradient(img, corner)
            refined_corners.append(refined)
        corners = np.array(refined_corners, dtype=np.int32)
    else:
        corners = corners.astype(np.int32)

    return corners


def cluster_lines_into_rulers(lines, img_shape, img=None, edges=None):
    """将检测到的线段聚类成两把尺子"""
    if lines is None or len(lines) == 0:
        return []

    vertical_lines = []
    horizontal_lines = []
    angle_distribution = []

    # 分类线段为垂直和水平
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # 计算线段的角度
        angle = np.arctan2(dy, dx) * 180 / np.pi
        angle_distribution.append(angle)

        # 垂直线：角度接近90度
        if angle > 60:
            vertical_lines.append(line[0])
        # 水平线：角度接近0度
        elif angle < 30:
            horizontal_lines.append(line[0])

    if angle_distribution:
        print(f"角度分布: 最小={min(angle_distribution):.1f}°, 最大={max(angle_distribution):.1f}°, 平均={np.mean(angle_distribution):.1f}°")

    rulers = []

    # 处理垂直线 - 从中心向外找边界
    print(f"检测到 {len(vertical_lines)} 条垂直线")
    if len(vertical_lines) > 0:
        vertical_lines = np.array(vertical_lines)

        # 计算每条线段的中点x坐标
        x_midpoints = (vertical_lines[:, 0] + vertical_lines[:, 2]) / 2
        y_midpoints = (vertical_lines[:, 1] + vertical_lines[:, 3]) / 2

        # 找到y方向的中心（图像中央）
        img_center_y = img_shape[0] / 2

        # 找到最接近图像中央的线段
        y_distances = np.abs(y_midpoints - img_center_y)
        center_line_idx = np.argmin(y_distances)
        center_x = x_midpoints[center_line_idx]

        print(f"垂直线中心x={center_x:.1f}")

        # 找到所有x坐标接近中心的线段（在中心±150像素范围内）
        x_tolerance = 150
        near_center_mask = np.abs(x_midpoints - center_x) <= x_tolerance
        near_center_lines = vertical_lines[near_center_mask]

        print(f"在中心±{x_tolerance}像素范围内找到 {len(near_center_lines)} 条垂直线")

        if len(near_center_lines) > 0:
            x_coords = np.concatenate([near_center_lines[:, 0], near_center_lines[:, 2]])
            y_coords = np.concatenate([near_center_lines[:, 1], near_center_lines[:, 3]])
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            print(f"垂直尺子: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], 宽={x_max-x_min}, 高={y_max-y_min}")

            # 垂直尺子应该有足够的高度
            if (y_max - y_min) > 50:
                # 从坐标中找到精确的四个角点
                corner_points = find_corners_from_coords(x_coords, y_coords, img=img, edges=edges)
                if corner_points is not None:
                    corners = corner_points
                else:
                    corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)

                rulers.append({
                    'type': 'vertical',
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'width': x_max - x_min,
                    'height': y_max - y_min,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'center': ((x_min + x_max) / 2, (y_min + y_max) / 2),
                    'corners': corners
                })
            else:
                print(f"垂直尺子高度不足: {y_max-y_min} <= 50")

    # 处理水平线 - 从中心向外找边界
    if len(horizontal_lines) > 0:
        horizontal_lines = np.array(horizontal_lines)

        # 计算每条线段的中点坐标
        x_midpoints = (horizontal_lines[:, 0] + horizontal_lines[:, 2]) / 2
        y_midpoints = (horizontal_lines[:, 1] + horizontal_lines[:, 3]) / 2

        # 找到x方向的中心（图像中央）
        img_center_x = img_shape[1] / 2

        # 找到最接近图像中央的线段
        x_distances = np.abs(x_midpoints - img_center_x)
        center_line_idx = np.argmin(x_distances)
        center_y = y_midpoints[center_line_idx]

        print(f"水平线中心y={center_y:.1f}")

        # 找到所有y坐标接近中心的线段（在中心±200像素范围内）
        y_tolerance = 200
        near_center_mask = np.abs(y_midpoints - center_y) <= y_tolerance
        near_center_lines = horizontal_lines[near_center_mask]

        if len(near_center_lines) > 0:
            x_coords = np.concatenate([near_center_lines[:, 0], near_center_lines[:, 2]])
            y_coords = np.concatenate([near_center_lines[:, 1], near_center_lines[:, 3]])
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            print(f"水平尺子: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], 宽={x_max-x_min}, 高={y_max-y_min}")

            # 水平尺子应该有足够的宽度
            if (x_max - x_min) > 100:
                # 从坐标中找到精确的四个角点
                corner_points = find_corners_from_coords(x_coords, y_coords, img=img, edges=edges)
                if corner_points is not None:
                    corners = corner_points
                else:
                    corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)

                rulers.append({
                    'type': 'horizontal',
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'width': x_max - x_min,
                    'height': y_max - y_min,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'center': ((x_min + x_max) / 2, (y_min + y_max) / 2),
                    'corners': corners
                })

    return rulers


def draw_ruler_info(img, rulers):
    """在图像上绘制尺子信息"""
    img_copy = img.copy()

    for i, ruler in enumerate(rulers):
        # 绘制矩形框
        corners = ruler['corners']
        cv2.drawContours(img_copy, [corners], 0, (0, 0, 255), 2)

        # 绘制角点
        for j, corner in enumerate(corners):
            cv2.circle(img_copy, tuple(corner), 5, (255, 0, 0), -1)
            cv2.putText(img_copy, f"C{j}", tuple(corner + np.array([5, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 绘制中心点
        center = tuple(map(int, ruler['center']))
        cv2.circle(img_copy, center, 3, (0, 255, 255), -1)

        # 添加文字信息
        y_offset = 30 + i * 120
        info_text = [
            f"Ruler {i+1} ({ruler['type']}):",
            f"Width: {ruler['width']:.2f} px",
            f"Height: {ruler['height']:.2f} px",
            f"Area: {ruler['area']:.2f} px2"
        ]

        for j, text in enumerate(info_text):
            cv2.putText(img_copy, text, (10, y_offset + j * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_copy


def analyze_ruler_image(image_path, calib_file, output_dir, calib_image_dir=None):
    """分析尺子图像"""
    print(f"处理图像: {image_path}")

    # 加载标定参数
    A, k = load_calibration_result(calib_file)
    print(f"相机内参:\n{A}")
    print(f"畸变系数: {k}\n")

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"原始图像尺寸: {w}x{h}")

    # 移除边缘阴影
    img_no_shadow = remove_shadow_edges(img, margin_percent=8)
    h_ns, w_ns = img_no_shadow.shape[:2]
    print(f"去除阴影后尺寸: {w_ns}x{h_ns}")

    # 畸变校正
    img_undistorted = undistort_image(img_no_shadow, A, k)
    print("畸变校正完成")

    # 移除黑边并记录偏移
    img_no_black = remove_black_borders(img_undistorted, threshold=20)
    h_nb, w_nb = img_no_black.shape[:2]
    print(f"去除黑边后尺寸: {w_nb}x{h_nb}")

    # 计算黑边的偏移量
    offset_y = (img_undistorted.shape[0] - h_nb) // 2
    offset_x = (img_undistorted.shape[1] - w_nb) // 2
    print(f"黑边偏移: x_offset={offset_x}, y_offset={offset_y}")

    # 尝试从当前图像计算转换比例，如果失败则从标定图像计算
    ratio_mm_per_px = calculate_pixel_to_mm_ratio(img_undistorted, square_size_mm=15, points_per_row=11, points_per_col=8)

    # 如果当前图像中没有棋盘格，尝试从标定图像目录中获取
    if ratio_mm_per_px is None and calib_image_dir is not None:
        print("尝试从标定图像中计算转换比例...")
        calib_files = [os.path.join(calib_image_dir, f) for f in os.listdir(calib_image_dir)
                       if f.lower().endswith(('.bmp', '.jpg', '.png'))]
        for calib_img_path in calib_files[:3]:  # 尝试前3张标定图像
            calib_img = cv2.imread(calib_img_path)
            if calib_img is not None:
                ratio_mm_per_px = calculate_pixel_to_mm_ratio(calib_img, square_size_mm=15, points_per_row=11, points_per_col=8)
                if ratio_mm_per_px is not None:
                    print(f"从标定图像 {os.path.basename(calib_img_path)} 中获得转换比例\n")
                    break

    # 检测尺子
    edges, lines = detect_rulers_with_edges(img_undistorted)
    print(f"检测到 {len(lines) if lines is not None else 0} 条线段")

    # 调试：统计垂直和水平线
    if lines is not None:
        vertical_count = 0
        horizontal_count = 0
        vertical_lines_debug = []
        horizontal_lines_debug = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if angle > 60:
                vertical_count += 1
                vertical_lines_debug.append((x1, y1, x2, y2, angle))
            elif angle < 30:
                horizontal_count += 1
                horizontal_lines_debug.append((x1, y1, x2, y2, angle))
        print(f"垂直线: {vertical_count}, 水平线: {horizontal_count}")

        # 打印前几条垂直线
        if vertical_lines_debug:
            print("垂直线样本:")
            for i, (x1, y1, x2, y2, angle) in enumerate(vertical_lines_debug[:3]):
                print(f"  线{i}: ({x1}, {y1}) -> ({x2}, {y2}), 角度={angle:.1f}°")

        # 打印前几条水平线
        if horizontal_lines_debug:
            print("水平线样本:")
            for i, (x1, y1, x2, y2, angle) in enumerate(horizontal_lines_debug[:3]):
                print(f"  线{i}: ({x1}, {y1}) -> ({x2}, {y2}), 角度={angle:.1f}°")

    # 聚类成尺子
    rulers = cluster_lines_into_rulers(lines, img_undistorted.shape, img=img_undistorted, edges=edges)
    print(f"识别到 {len(rulers)} 把尺子\n")

    if len(rulers) == 0:
        print("未检测到尺子")
        return

    # 如果还没有获得转换比例，再尝试一次从当前图像计算
    if ratio_mm_per_px is None:
        ratio_mm_per_px = calculate_pixel_to_mm_ratio(img_undistorted, square_size_mm=15, points_per_row=11, points_per_col=8)

    # 分析每个尺子
    for i, ruler in enumerate(rulers):
        print(f"尺子 {i+1} ({ruler['type']}):")
        print(f"  长: {ruler['width']:.2f} px", end="")
        if ratio_mm_per_px is not None:
            print(f" ({ruler['width'] * ratio_mm_per_px:.2f} mm)")
        else:
            print()

        print(f"  宽: {ruler['height']:.2f} px", end="")
        if ratio_mm_per_px is not None:
            print(f" ({ruler['height'] * ratio_mm_per_px:.2f} mm)")
        else:
            print()

        print(f"  面积: {ruler['area']:.2f} px2", end="")
        if ratio_mm_per_px is not None:
            print(f" ({ruler['area'] * ratio_mm_per_px**2:.2f} mm2)")
        else:
            print()

        print(f"  中心: ({ruler['center'][0]:.2f}, {ruler['center'][1]:.2f})")
        print(f"  角点: {ruler['corners']}\n")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)

    # 保存原始图像
    cv2.imwrite(os.path.join(output_dir, "01_original.jpg"), img)

    # 保存校正后的图像
    cv2.imwrite(os.path.join(output_dir, "02_undistorted.jpg"), img_undistorted)

    # 保存边缘检测结果
    cv2.imwrite(os.path.join(output_dir, "03_edges.jpg"), edges)

    # 保存标注结果
    result_img = draw_ruler_info(img_undistorted, rulers)
    cv2.imwrite(os.path.join(output_dir, "04_ruler_analysis.jpg"), result_img)

    print(f"结果已保存到: {output_dir}")

    # 生成报告
    report_path = os.path.join(output_dir, "ruler_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("尺子分析报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"图像: {image_path}\n")
        f.write(f"原始尺寸: {w}x{h}\n")
        if ratio_mm_per_px is not None:
            f.write(f"转换比例: {ratio_mm_per_px:.4f} mm/px\n")
        f.write("\n")

        for i, ruler in enumerate(rulers):
            f.write(f"尺子 {i+1} ({ruler['type']}):\n")
            f.write(f"  长: {ruler['width']:.2f} px")
            if ratio_mm_per_px is not None:
                f.write(f" ({ruler['width'] * ratio_mm_per_px:.2f} mm)")
            f.write("\n")

            f.write(f"  宽: {ruler['height']:.2f} px")
            if ratio_mm_per_px is not None:
                f.write(f" ({ruler['height'] * ratio_mm_per_px:.2f} mm)")
            f.write("\n")

            f.write(f"  面积: {ruler['area']:.2f} px2")
            if ratio_mm_per_px is not None:
                f.write(f" ({ruler['area'] * ratio_mm_per_px**2:.2f} mm2)")
            f.write("\n")

            f.write(f"  中心: ({ruler['center'][0]:.2f}, {ruler['center'][1]:.2f})\n")
            f.write(f"  角点:\n")
            for j, corner in enumerate(ruler['corners']):
                f.write(f"    C{j}: ({corner[0]}, {corner[1]})\n")
            f.write("\n")

    print(f"报告已保存到: {report_path}")


def main():
    # 路径配置
    image_path = r"D:\vision\mv\zhangzhenyou\used data\rule_test\b.bmp"
    calib_file = r"D:\vision\mv\zhangzhenyou\output\opencv.txt"
    output_dir = r"D:\vision\mv\zhangzhenyou\output\ruler_analysis"
    calib_image_dir = r"D:\vision\mv\zhangzhenyou\used data\img"  # 标定图像目录

    analyze_ruler_image(image_path, calib_file, output_dir, calib_image_dir)


if __name__ == '__main__':
    main()

