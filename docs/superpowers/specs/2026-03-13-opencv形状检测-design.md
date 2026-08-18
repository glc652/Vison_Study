# OpenCV 多图片形状检测设计文档

**日期**：2026-03-13
**目标**：在四张图片（2-1.jpg ~ 2-4.jpg）中识别三角形、矩形、圆形，并输出坐标位置

## 1. 需求概述

- 输入：4张图片，尺寸约 640x480
- 输出：每张图片中所有形状的类型和坐标位置
- 图片问题：模糊、反光、阴影干扰
- 形状类型：三角形、矩形、圆形（数量不固定）

## 2. 整体架构

```
┌─────────────────────────────────────────┐
│              图像预处理                  │
│  (对比度增强、去噪、形态学处理)            │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│           边缘检测与融合                  │
│    (Canny、Sobel、Laplacian 多尺度)       │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│           轮廓提取与过滤                  │
│       (面积筛选、形态学填充)              │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│           形状分类与坐标计算              │
│   (顶点数量、圆度、矩形度等特征判断)       │
└─────────────────────────────────────────┘
```

## 3. 图像预处理模块

### 3.1 对比度增强
```python
# 使用 CLAHE 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)
```

### 3.2 去模糊处理
```python
# 针对模糊图片使用锐化卷积核
kernel_sharpen = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
```

### 3.3 去反光处理
```python
# 使用形态学 Top-hat 提取反光区域
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
# 从原图中减去反光区域
result = cv2.subtract(enhanced, tophat)
```

### 3.4 阴影处理
```python
# 使用色彩空间分离阴影
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
v_channel = hsv[:,:,2]
# 形态学闭运算填充阴影区域
kernel = np.ones((21,21), np.uint8)
closed = cv2.morphologyEx(v_channel, cv2.MORPH_CLOSE, kernel)
```

### 3.5 自适应预处理流程
```python
def adaptive_preprocess(image_path):
    """根据图像质量自动选择预处理方法"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像质量指标
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 步骤1：对比度增强（所有图片都需要）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    result = clahe.apply(gray)

    # 步骤2：如果模糊，进行锐化
    if blur_score < 100:
        kernel_sharpen = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        result = cv2.filter2D(result, -1, kernel_sharpen)

    # 步骤3：去反光
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    tophat = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel)
    result = cv2.subtract(result, tophat)

    # 步骤4：形态学闭运算处理阴影
    kernel = np.ones((11,11), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return img, result
```

## 4. 边缘检测模块

### 4.1 多尺度边缘融合
```python
def multi_scale_edge_detection(gray):
    # 多尺度 Canny
    edges1 = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
    edges2 = cv2.Canny(cv2.GaussianBlur(gray, (9,9), 0), 30, 100)

    # Sobel 梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
    _, edges3 = cv2.threshold(sobel_mag.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)

    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_mag = cv2.convertScaleAbs(lap)
    _, edges4 = cv2.threshold(lap_mag, 30, 255, cv2.THRESH_BINARY)

    # 边缘融合
    combined = cv2.bitwise_or(edges1, edges2)
    combined = cv2.bitwise_or(combined, edges3)
    combined = cv2.bitwise_or(combined, edges4)

    return combined
```

### 4.2 自适应阈值调整
```python
def adaptive_canny_threshold(gray):
    """根据图像统计信息自动调整 Canny 阈值"""
    median = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return lower, upper
```

## 5. 形态学处理模块

### 5.1 边缘连接与填充
```python
def morphology_process(edges):
    # 闭运算连接断裂边缘
    kernel = np.ones((9,9), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)

    # 膨胀连接相邻边缘
    dilated = cv2.dilate(closed, kernel, iterations=3)

    return dilated
```

### 5.2 轮廓填充与空洞处理
```python
def fill_and_process_contours(binary):
    """填充轮廓并处理内部空洞"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(binary)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 15000:  # 过滤小轮廓
            cv2.drawContours(filled, [contour], -1, 255, -1)

    # 形态学闭运算填充空洞
    kernel = np.ones((21,21), np.uint8)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=3)

    return filled
```

## 6. 形状分类模块

### 6.1 特征计算
```python
def calculate_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # 圆度
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    # 矩形特征
    rect = cv2.minAreaRect(contour)
    rect_area = rect[1][0] * rect[1][1]
    rect_fill_ratio = area / rect_area if rect_area > 0 else 0

    # 长宽比
    aspect_ratio = max(rect[1]) / min(rect[1]) if min(rect[1]) > 0 else 1

    # 顶点数量
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'rect_fill_ratio': rect_fill_ratio,
        'aspect_ratio': aspect_ratio,
        'vertices': vertices
    }
```

### 6.2 形状判断逻辑
```python
def classify_shape(features):
    c = features['circularity']
    v = features['vertices']
    r = features['rect_fill_ratio']
    a = features['aspect_ratio']

    # 圆形：高圆度
    if c > 0.7:
        return '圆形', 0.9

    # 三角形：3个顶点
    if v == 3:
        return '三角形', 0.9

    # 矩形：4个顶点且填充率高，或长宽比大
    if v == 4 and r > 0.6:
        return '矩形', 0.9
    if a > 2.5:
        return '矩形', 0.85

    # 其他情况
    if v <= 6 and r < 0.65:
        return '三角形', 0.7

    return '未知', 0.5
```

## 7. 输出格式

### 7.1 位置信息
```python
{
    'name': '矩形',           # 形状名称
    'confidence': 0.9,        # 置信度
    'position': (x, y, w, h), # 边界框 (左上角坐标 + 宽高)
    'center': (cx, cy),       # 中心点坐标
    'area': 50000,            # 面积
    'contour': contour        # 轮廓点集
}
```

### 7.2 终端输出示例
```
【矩形】数量：1
  1. 位置：(100, 50), 宽=200, 高=150
     中心点：(200, 125), 面积：30000

【圆形】数量：1
  1. 位置：(350, 200), 宽=100, 高=100
     中心点：(400, 250), 面积：7854
```

## 8. 文件结构

```
mv/
├── test3.py              # 主程序入口
├── preprocess/           # 预处理模块
│   └── __init__.py
│   └── adaptive.py       # 自适应预处理
├── edge/                 # 边缘检测模块
│   ├── __init__.py
│   └── detection.py      # 多尺度边缘检测
├── morphology/           # 形态学处理模块
│   ├── __init__.py
│   └── process.py        # 形态学处理
├── shape/                # 形状分类模块
│   ├── __init__.py
│   └── classifier.py     # 形状分类
└── utils/                # 工具函数
    ├── __init__.py
    └── visualizer.py     # 可视化工具
```

## 9. 测试计划

| 图片 | 预期形状 | 重点测试 |
|------|----------|----------|
| 2-1.jpg | 三角形、矩形、圆形 | 基准测试 |
| 2-2.jpg | 三角形、矩形、圆形 | 反光处理 |
| 2-3.jpg | 三角形、矩形、圆形 | 阴影处理 |
| 2-4.jpg | 三角形、矩形、圆形 | 模糊+反光综合处理 |

## 10. 后续优化

1. **参数调优**：根据实际运行效果调整阈值和核大小
2. **添加新形状**：扩展 classify_shape 函数支持更多形状
3. **批量处理**：支持一次处理多张图片
4. **性能优化**：对实时性要求场景进行加速