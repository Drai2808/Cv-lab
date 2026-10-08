import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
def load_and_split(dataset_path, test_size):
    images = []
    labels = []
    for label_name in ['normal', 'abnormal']:
        folder = os.path.join(dataset_path, label_name)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, fname))
                if img is not None:
                    images.append(img)
                    labels.append(1 if label_name == 'abnormal' else 0)
    if len(images) == 0:
        raise ValueError("No images found. Ensure 'normal' and 'abnormal' folders exist in the dataset path.")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42, stratify=labels, shuffle=True)
    return X_train, X_test, y_train, y_test