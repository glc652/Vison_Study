# import cv2
# import numpy as np

# # 1. 读取图片
# img = cv2.imread(r"D:\vision\mv\yolo\bike\images\test\1.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # 2. 高斯模糊，降低噪声
# gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

# # 3. Canny 边缘检测
# edges = cv2.Canny(gray_blur, 50, 150)

# # 4. Harris 角点检测
# gray_float = np.float32(gray)
# dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

# # 5. 膨胀角点，使其可见
# dst = cv2.dilate(dst, None)

# # 6. 标记角点
# img_copy = img.copy()
# img_copy[dst > 0.01 * dst.max()] = [0, 0, 255]  # 红色角点

# # 7. 可视化
# cv2.imshow("Edges", edges)
# cv2.imshow("Corners", img_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# from ultralytics import YOLO
# import cv2
# import numpy as np

# # 加载模型
# model = YOLO('bike/runs/detect/train/weights/best.pt')

# # 预测单张图片
# image_path = r"D:\vision\mv\yolo\bike\images\test\1.png"
# results = model.predict(source=image_path, save=False, conf=0.04)

# # 读取原始图片
# image = cv2.imread(image_path)

# # 存储所有中心点
# centers = []

# # 处理结果并绘制框
# for r in results:
#     boxes = r.boxes  # 边界框对象
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 获取坐标
#         cls = int(box.cls)                                # 获取类别ID
#         conf = float(box.conf)                            # 获取置信度

#         # 获取类别名称
#         class_name = model.names[cls]

#         # 计算框的宽度和高度
#         box_width = x2 - x1
#         box_height = y2 - y1

#         # 计算框的中心点
#         center_x = (x1 + x2) // 2
#         center_y = (y1 + y2) // 2

#         # 保存中心点信息
#         centers.append({
#             'point': (center_x, center_y),
#             'class_name': class_name,
#             'conf': conf,
#             'box': (x1, y1, x2, y2)
#         })

#         print(f"Class: {class_name}, Confidence: {conf:.2%}, Box: [{x1}, {y1}, {x2}, {y2}]")
#         print(f"  框大小: {box_width}x{box_height} 像素")
#         print(f"  中心点: ({center_x}, {center_y})\n")

#         # 绘制矩形框
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # 绘制中心点（红色圆点）
#         cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

#         # 绘制中心点坐标标签
#         cv2.putText(image, f"({center_x}, {center_y})", (center_x + 10, center_y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# # 如果有多个中心点，连接相邻的中心点
# if len(centers) > 1:
#     print(f"\n检测到 {len(centers)} 个目标，计算相邻中心点距离:\n")

#     # 按x坐标排序中心点
#     centers_sorted = sorted(centers, key=lambda c: c['point'][0])

#     # 连接相邻的中心点并计算距离
#     for i in range(len(centers_sorted) - 1):
#         p1 = centers_sorted[i]['point']
#         p2 = centers_sorted[i + 1]['point']

#         # 计算欧几里得距离
#         distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

#         print(f"中心点 {i+1} -> {i+2}: {p1} -> {p2}, 距离: {distance:.2f} 像素")

#         # 绘制连接线
#         cv2.line(image, p1, p2, (255, 255, 0), 2)

#         # 在连接线中点显示距离
#         mid_x = (p1[0] + p2[0]) // 2
#         mid_y = (p1[1] + p2[1]) // 2
#         cv2.putText(image, f"{distance:.1f}px", (mid_x, mid_y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


# # 显示图片
# cv2.imshow("Detection Result", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 保存标注后的图片
# output_path = r"D:\vision\mv\yolo\output_annotated.jpg"
# cv2.imwrite(output_path, image)
# print(f"\n标注图片已保存: {output_path}")
