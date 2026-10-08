from skimage.feature import hog
import numpy as np
from processing import preprocess, detect_edges
def extract_features(edge_img):
    fd, _ = hog(edge_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return fd
def extract_batch(images, low, high):
    features = []
    for img in images:
        gray = preprocess(img)
        edges = detect_edges(gray, low, high)
        fd = extract_features(edges)
        features.append(fd)
    return np.array(features)