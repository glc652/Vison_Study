'''
SVM学习

   """
        训练SVM模型

        参数:
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
            C: 正则化参数
            gamma: 核函数系数
            grid_search: 是否进行网格搜索调参
        """
      param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
'''

import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle

def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize image to a fixed size (e.g., 64x64)
            resized_img = cv2.resize(img, (64, 64))
            # Flatten the image into a 1D array
            flattened_img = resized_img.flatten()
            images.append(flattened_img)
            labels.append(label)
    return images, labels

def main():
    # Define paths to your dataset folders
    # folder_cat = "image/bolt"
    # folder_dog = "image/locatingpin"
    folder_bolt = "D:\\vision\\blnw-images\\blnw-images-224\\bolt"
    folder_locatingpin = "D:\\vision\\blnw-images\\blnw-images-224\\locatingpin"

    # Load images and their corresponding labels
    bolt_images, bolt_labels = load_images_from_folder(folder_bolt, 0)  # Label cats as 0
    locatingpin_images, locatingpin_labels = load_images_from_folder(folder_locatingpin, 1)  # Label dogs as 1

    # Combine data from both classes
    X = np.array(bolt_images + locatingpin_images)
    y = np.array(bolt_labels + locatingpin_labels)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Create SVM classifier with RBF kernel
    clf = svm.SVC(kernel='rbf', gamma='scale')

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Save trained model to disk
    save_path = r"D:\vision\mv\SVM\models\svm_model.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)

    print(f"Trained model saved as {save_path}.")

if __name__ == "__main__":
    main()
