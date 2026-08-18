import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 加载图像数据
def load_images(root_dir, image_size=(32, 32)):
    X_list = []
    y_list = []
    class_names = []
    root = Path(root_dir)

    print(f"正在从 '{root}' 加载图片...")
    for cls_idx, cls_name in enumerate(sorted([p.name for p in root.iterdir() if p.is_dir()])):
        class_names.append(cls_name)
        cls_dir = root / cls_name
        image_count = 0
        for img_path in cls_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            try:
                with Image.open(img_path) as im:
                    im = im.convert("L").resize(image_size)
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                    X_list.append(arr.flatten())
                    y_list.append(cls_idx)
                    image_count += 1
            except Exception:
                continue
        print(f"  - 类别 '{cls_name}': {image_count} 张图片")

    x = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=int)
    return x, y, class_names

# 加载数据
x, y, class_names = load_images(r'D:\vision\blnw-images\blnw-images-224', image_size=(32, 32))
print(f"✓ 加载完成。类别: {class_names}，总样本数: {len(x)}\n")

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=None)

# 标准化
std = StandardScaler()
x_train_standard = std.fit_transform(x_train)
x_test_standard = std.transform(x_test)

# 训练KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_standard, y_train)
y_predict = knn.predict(x_test_standard)

# 输出结果
acc = knn.score(x_test_standard, y_test)
print(f"准确率: {acc:.4f}\n")
print("分类报告:")
print(classification_report(y_test, y_predict, target_names=class_names))
