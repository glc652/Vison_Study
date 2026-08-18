import numpy as np
import cv2
import os
from pathlib import Path
from knn_classifier import KNNClassifier, train_test_split_indices

# 图片数据集路径
dataset_path = r"D:\vision\blnw-images\blnw-images-224"
classes = ["bolt", "locatingpin", "nut", "washer"]

# 加载图片并转换为特征向量
X = []
y = []
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 将图片展平为一维向量
                features = img.flatten().astype(np.float32)
                X.append(features)
                y.append(class_to_idx[class_name])

X = np.array(X)
y = np.array(y)

print(f"数据集大小: {X.shape}")
print(f"类别分布: {np.bincount(y)}")

# 拆分训练和测试集
train_idx, test_idx = train_test_split_indices(len(X), test_ratio=0.3, rng=np.random.default_rng(42))
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 训练 KNN 分类器
clf = KNNClassifier(p=2)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test, k=5)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print(f"\n准确率: {accuracy:.2%}")

# 显示前10个预测结果
print(f"\n前10个预测结果:")
for i in range(min(10, len(y_test))):
    pred_class = classes[y_pred[i]]
    true_class = classes[y_test[i]]
    match = "✓" if y_pred[i] == y_test[i] else "✗"
    print(f"{match} 预测: {pred_class:12} | 真实: {true_class:12}")
