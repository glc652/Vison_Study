from ultralytics import YOLO
import cv2
import numpy as np

# 加载模型
model = YOLO('bike/runs/detect/train/weights/best.pt')

# ===== 单目测距参数 =====
# 相机内参（需要通过标定获得）
FOCAL_LENGTH = 800         # 焦距（像素）
PRINCIPAL_POINT_X = 320    # 主点x坐标（图片宽度/2）
PRINCIPAL_POINT_Y = 243    # 主点y坐标（图片高度/2）

# 已知物体尺寸
BIKE_REAL_HEIGHT = 0.64     # 自行车实际高度（米）
BIKE_REAL_WIDTH = 1.7      # 自行车实际宽度（米）

# 相机高度（假设相机安装高度）
CAMERA_HEIGHT = 1.5        # 相机离地面高度（米）

class MonocularDistanceEstimator:
    """单目测距估算器"""

    def __init__(self, focal_length, principal_point_x, principal_point_y, camera_height):
        self.f = focal_length
        self.cx = principal_point_x
        self.cy = principal_point_y
        self.camera_height = camera_height

    def estimate_distance_by_height(self, box_height_pixels, box_y_center):
        """
        基于物体高度和图像中的位置估算距离

        Args:
            box_height_pixels: 检测框高度（像素）
            box_y_center: 检测框中心y坐标（像素）

        Returns:
            距离（米）
        """
        if box_height_pixels == 0:
            return float('inf')

        # 方法1: 基于物体高度的投影
        # 物体在图像上的高度 = 焦距 * 物体实际高度 / 距离
        # 因此: 距离 = 焦距 * 物体实际高度 / 物体在图像上的高度
        distance = (self.f * BIKE_REAL_HEIGHT) / box_height_pixels

        return distance

    def estimate_distance_by_position(self, box_y_bottom, box_height_pixels):
        """
        基于物体在图像中的位置估算距离（假设物体在地面上）

        Args:
            box_y_bottom: 检测框底部y坐标（像素）
            box_height_pixels: 检测框高度（像素）

        Returns:
            距离（米）
        """
        if box_height_pixels == 0:
            return float('inf')

        # 物体底部到主点的距离（像素）
        dy = box_y_bottom - self.cy

        # 使用相似三角形原理
        # tan(θ) = dy / f
        # 物体距离 = 相机高度 / tan(θ) = 相机高度 * f / dy
        if dy == 0:
            return float('inf')

        distance = (self.camera_height * self.f) / dy

        return distance

    def estimate_distance_combined(self, box_height_pixels, box_y_bottom):
        """
        基于物体高度估算距离

        Args:
            box_height_pixels: 检测框高度（像素）
            box_y_bottom: 检测框底部y坐标（像素）

        Returns:
            距离（米）
        """
        distance = self.estimate_distance_by_height(box_height_pixels, box_y_bottom)
        return distance, distance, distance

def estimate_real_distance_between_bikes(pixel_distance, dist1, dist2):
    """
    估算两个自行车之间的真实距离

    Args:
        pixel_distance: 中心点之间的像素距离
        dist1, dist2: 两个自行车到相机的距离（米）

    Returns:
        真实距离（米）
    """
    avg_dist = (dist1 + dist2) / 2
    real_distance = (pixel_distance * avg_dist) / FOCAL_LENGTH
    return real_distance

# 初始化单目测距估算器
estimator = MonocularDistanceEstimator(FOCAL_LENGTH, PRINCIPAL_POINT_X, PRINCIPAL_POINT_Y, CAMERA_HEIGHT)

# 预测单张图片
image_path = r"D:\vision\mv\yolo\bike\images\test\1.png"
results = model.predict(source=image_path, save=False, conf=0.04)

# 读取原始图片
image = cv2.imread(image_path)
img_height, img_width = image.shape[:2]

# 更新主点坐标
estimator.cx = img_width / 2
estimator.cy = img_height / 2

# 存储所有中心点
centers = []

# 处理结果并绘制框
for r in results:
    boxes = r.boxes  # 边界框对象
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 获取坐标
        cls = int(box.cls)                                # 获取类别ID
        conf = float(box.conf)                            # 获取置信度

        # 获取类别名称
        class_name = model.names[cls]

        # 计算框的宽度和高度
        box_width = x2 - x1
        box_height = y2 - y1

        # 计算框的中心点和底部
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_y_bottom = y2

        # 使用单目测距估算距离
        combined_dist, dist_by_height, dist_by_position = estimator.estimate_distance_combined(box_height, box_y_bottom)

        # 保存中心点信息
        centers.append({
            'point': (center_x, center_y),
            'class_name': class_name,
            'conf': conf,
            'box': (x1, y1, x2, y2),
            'box_height': box_height,
            'distance_to_camera': combined_dist,
            'dist_by_height': dist_by_height,
            'dist_by_position': dist_by_position
        })

        print(f"Class: {class_name}, Confidence: {conf:.2%}")
        print(f"  框: [{x1}, {y1}, {x2}, {y2}], 高度: {box_height} 像素")
        print(f"  估算距离: {combined_dist:.2f} 米")
        print(f"  中心点: ({center_x}, {center_y})\n")

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制中心点（红色圆点）
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

        # 绘制距离标签
        label = f"{class_name} {conf:.2%} | {combined_dist:.2f}m"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 如果有多个中心点，连接相邻的中心点
if len(centers) > 1:
    print(f"\n检测到 {len(centers)} 个目标，计算相邻中心点距离:\n")

    # 按x坐标排序中心点
    centers_sorted = sorted(centers, key=lambda c: c['point'][0])

    # 连接相邻的中心点并计算距离
    for i in range(len(centers_sorted) - 1):
        p1 = centers_sorted[i]['point']
        p2 = centers_sorted[i + 1]['point']
        dist1 = centers_sorted[i]['distance_to_camera']
        dist2 = centers_sorted[i + 1]['distance_to_camera']

        # 计算像素距离
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        # 计算真实距离
        real_distance = estimate_real_distance_between_bikes(pixel_distance, dist1, dist2)

        print(f"自行车 {i+1} -> {i+2}:")
        print(f"  像素距离: {pixel_distance:.2f} px")
        print(f"  到相机距离: {dist1:.2f}m, {dist2:.2f}m")
        print(f"  估算真实距离: {real_distance:.2f} 米\n")

        # 绘制连接线
        cv2.line(image, p1, p2, (255, 255, 0), 2)

        # 在连接线中点显示距离
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(image, f"{real_distance:.2f}m", (mid_x, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# 显示图片
cv2.imshow("Monocular Distance Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存标注后的图片
output_path = r"D:\vision\mv\yolo\output_annotated.jpg"
cv2.imwrite(output_path, image)
print(f"\n标注图片已保存: {output_path}")
