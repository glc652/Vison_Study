import cv2
import numpy as np
import pickle
import os
import pandas as pd


def preprocess_image(image_path):
    """
    对输入图像进行预处理，使其适用于模型预测。

    该函数读取指定路径的灰度图像，将其调整为固定尺寸（64x64），展平为一维数组，
    并对像素值进行归一化（除以255.0），最后重塑为适用于单样本预测的二维数组格式。

    参数:
        image_path (str): 输入图像文件的路径。

    返回:
        numpy.ndarray: 形状为 (1, 4096) 的归一化图像数组，其中 4096 = 64 * 64。

    异常:
        ValueError: 当无法读取指定路径的图像文件时抛出。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized_img = cv2.resize(img, (64, 64))
        flattened_img = resized_img.flatten()
        normalized_img = flattened_img / 255.0  # Normalize pixel values
        return normalized_img.reshape(1, -1)  # Reshape for single sample
    else:
        raise ValueError("Could not read image file.")


def predict_class(model_path, image_path):
    """
    使用预训练的SVM模型对给定图像进行类别预测。

    该函数加载保存的SVM模型，对输入图像进行预处理，并执行预测。
    预测结果为数值标签，随后映射回对应的类别名称。

    参数:
        model_path (str): 预训练SVM模型文件（pickle格式）的路径。
        image_path (str): 待预测图像文件的路径。

    返回:
        str: 预测的类别名称，当前支持的类别为 'bolt' 或 'locatingpin'。

    异常:
        可能抛出与文件读取、模型加载或图像处理相关的异常（如FileNotFoundError、ValueError等）。
    """
    # Load trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Preprocess input image
    processed_img = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(processed_img)

    # Map numeric label back to original category name
    categories = ['bolt', 'locatingpin']
    predicted_category = categories[int(prediction)]

    return predicted_category


def batch_predict(model_path, folder_path, max_images=30):
    """批量预测文件夹中的图像，返回结果列表"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    categories = ['bolt', 'locatingpin']
    results = []

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for filename in sorted(image_files)[:max_images]:
        image_path = os.path.join(folder_path, filename)
        try:
            processed_img = preprocess_image(image_path)
            prediction = model.predict(processed_img)
            predicted_category = categories[int(prediction)]
            results.append({
                filename,
                predicted_category,
                # 'label': int(prediction)
            })
        except Exception as e:
            results.append({
                'filename': filename,
                'prediction': 'error',
                'label': -1,
                'error': str(e)
            })

    return results


# Example usage when running this script directly
if __name__ == "__main__":
    MODEL_PATH = r"D:\vision\mv\SVM\models\svm_model.pkl"
    FOLDER_PATH = r"D:\vision\blnw-images\blnw-images-224\locatingpin"

    try:
        results = batch_predict(MODEL_PATH, FOLDER_PATH, max_images=30)
        print(results)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")