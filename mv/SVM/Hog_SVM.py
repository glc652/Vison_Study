'''
HOG SVM
'''

import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle
from typing import List, Tuple
import pandas as pd


class HOGSVMClassifier:
    def __init__(self, win_size=(64, 128)):
        """
        初始化HOG+SVM分类器

        Args:
            win_size: HOG窗口大小，默认(64, 128)
        """
        self.svm_model = None
        self.categories = []
        self.win_size = win_size
        self.hog = cv2.HOGDescriptor(
            _winSize=win_size, #检测窗口大小
            _blockSize=(16, 16),  #块大小
            _blockStride=(8, 8),  #块步长
            _cellSize=(8, 8),  #单元大小
            _nbins=9  # 区间个数
        )

    def extract_hog_features(self, image_path: str) -> np.ndarray:
        """提取图像的HOG特征"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 调整图像大小
        resized = cv2.resize(gray, self.win_size)

        # 计算HOG特征
        hog_features = self.hog.compute(resized)

        return hog_features.flatten()

    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据集并提取HOG特征"""
        features = []
        labels = []
        self.categories = []

        # 遍历数据集目录
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path):
                continue

            if category not in self.categories:
                self.categories.append(category)

            print(f"处理类别: {category}")

            # 遍历类别目录中的图像
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(category_path, filename)
                    try:
                        # 提取HOG特征
                        feature_vector = self.extract_hog_features(image_path)
                        features.append(feature_vector)
                        labels.append(self.categories.index(category))
                    except Exception as e:
                        print(f"处理图像 {image_path} 时出错: {e}")

        return np.array(features), np.array(labels)

    def train(self, dataset_path: str) -> float:
        """训练SVM分类器"""
        print("加载数据集...")
        X, y = self.load_dataset(dataset_path)

        if len(X) == 0:
            raise ValueError("数据集中没有有效的图像")

        print(f"数据集大小: {len(X)} 个样本")
        print(f"特征维度: {X.shape}")
        print(f"类别数量: {len(self.categories)}")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 创建并训练SVM模型
        print("训练SVM分类器...")
        self.svm_model = svm.SVC(kernel='rbf', gamma='scale', probability=True)
        self.svm_model.fit(X_train, y_train)

        # 在测试集上评估
        y_pred = self.svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"测试准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=self.categories))

        return accuracy

    def predict(self, image_path: str) -> Tuple[str, float]:
        """预测单张图像的类别"""
        if self.svm_model is None:
            raise ValueError("模型尚未训练，请先调用train方法")

        # 提取HOG特征
        features = self.extract_hog_features(image_path)
        features = features.reshape(1, -1)  # 重塑为单个样本

        # 预测
        prediction = self.svm_model.predict(features)
        probabilities = self.svm_model.predict_proba(features)
        confidence = np.max(probabilities)

        # 返回类别名称和置信度
        category = self.categories[int(prediction)]
        return category, confidence

    def save_model(self, model_path: str):
        """保存训练好的模型"""
        if self.svm_model is None:
            raise ValueError("没有可保存的模型")

        model_data = {
            'svm_model': self.svm_model,
            'categories': self.categories,
            'win_size': self.win_size
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到: {model_path}")

    def load_model(self, model_path: str):
        """加载预训练模型"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.svm_model = model_data['svm_model']
        self.categories = model_data['categories']
        self.win_size = model_data['win_size']
        # 重新初始化HOG描述符
        self.hog = cv2.HOGDescriptor(
            _winSize=self.win_size,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        print(f"模型已从 {model_path} 加载")

    def test_folder(self, folder_path: str, max_images: int = 30) -> List[dict]:
        """测试文件夹中的前N张图片，返回结果列表"""
        if self.svm_model is None:
            raise ValueError("模型尚未训练，请先调用train方法或load_model方法")

        results = []
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        for filename in sorted(image_files)[:max_images]:
            image_path = os.path.join(folder_path, filename)
            try:
                category, confidence = self.predict(image_path)
                results.append({
                    'filename': filename,
                    'prediction': category,
                    'confidence': float(confidence)
                })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': str(e)
                })

        return results


def main():
    # 创建分类器实例
    classifier = HOGSVMClassifier(win_size=(64, 128))

    # # 训练模型（需要提供数据集路径）
    # # 注意：您需要准备一个包含子目录的数据集，每个子目录代表一个类别
    # # 例如: dataset/person/, dataset/car/
    # dataset_path = "image"  # 修改为您的数据集路径

    # if os.path.exists(dataset_path):
    #     try:
    #         accuracy = classifier.train(dataset_path)
    #         print(f"\n训练完成，准确率: {accuracy:.4f}")

    #         # 保存模型
    #         classifier.save_model(r"D:\vision\mv\SVM\models\hog_svm_model.pkl")

    #         # 示例预测（需要提供测试图像路径）
    #         # test_image_path = "test_image.jpg"
    #         # if os.path.exists(test_image_path):
    #         #     category, confidence = classifier.predict(test_image_path)
    #         #     print(f"\n预测结果: {category} (置信度: {confidence:.4f})")

    #     except Exception as e:
    #         print(f"训练过程中出错: {e}")
    # else:
    #     print(f"数据集路径不存在: {dataset_path}")
    #     print("请创建数据集目录并按类别组织图像文件")
    #     print("目录结构示例:")
    #     print("dataset/")
    #     print("  ├── person/")
    #     print("  │   ├── person1.jpg")
    #     print("  │   └── person2.jpg")
    #     print("  └── car/")
    #     print("      ├── car1.jpg")
    #     print("      └── car2.jpg")

    # 加载已训练的模型
    model_path = r"D:\vision\mv\SVM\models\hog_svm_model.pkl"

    try:
        classifier.load_model(model_path)

        # 测试两个文件夹
        bolt_folder = r"D:\vision\blnw-images\image\bolt"
        locatingpin_folder = r"D:\vision\blnw-images\image\locatingpin"

        print("测试bolt文件夹...")
        bolt_results = classifier.test_folder(bolt_folder, max_images=30)

        print("测试locatingpin文件夹...")
        locatingpin_results = classifier.test_folder(locatingpin_folder, max_images=30)

        # 合并结果
        all_results = []
        for result in bolt_results:
            result['folder'] = 'bolt'
            all_results.append(result)
        for result in locatingpin_results:
            result['folder'] = 'locatingpin'
            all_results.append(result)

        # 转换为DataFrame
        df = pd.DataFrame(all_results)

        # 创建输出目录
        output_dir = r"D:\vision\mv\SVM\tests"
        os.makedirs(output_dir, exist_ok=True)

        # 保存到Excel
        output_path = os.path.join(output_dir, "hog_svm_test_results.xlsx")
        df.to_excel(output_path, index=False)

        print(f"\n结果已保存到: {output_path}")
        print(f"总测试图片数: {len(all_results)}")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
