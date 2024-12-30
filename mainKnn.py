import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog, local_binary_pattern
import joblib

# Path to the IrisFlowerImages folder
base_path = os.path.expanduser("/Users/laura/Desktop/IrisFlowerImages")

# Define the subfolders (class names)
class_names = ['iris-setosa', 'iris-versicolour', 'iris-virginica']

def feature_extraction(image):
    # Segmentation 
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 25, 25])  # Lower bound of green
    upper_green = np.array([90, 255, 255])  # Upper bound of green

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)
    segmented_image = cv2.bitwise_and(image, image, mask=non_green_mask)

    # Grayscale conversion
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Contrast Stretching
    smin, smax = 0, 255
    cutoff_fraction = 0.02
    total_pixels = gray_image.size
    num_ignore = int(total_pixels * cutoff_fraction)

    hist, _ = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])
    lower_cutoff = np.searchsorted(np.cumsum(hist), num_ignore)
    upper_cutoff = 255 - np.searchsorted(np.cumsum(hist[::-1]), num_ignore)
    img_clipped = np.clip(gray_image, lower_cutoff, upper_cutoff)

    if upper_cutoff - lower_cutoff == 0:
        upper_cutoff += 1

    stretched = ((img_clipped - lower_cutoff) * ((smax - smin) / (upper_cutoff - lower_cutoff)) + smin).astype(np.uint8)

    # Canny edge detection
    edges = cv2.Canny(stretched, 50, 150)  

    # Feature extraction
    hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
    # HOG captures texture and shape information
    fd, _ = hog(edges, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    # LBP captures texture by encoding the relationships between a pixel and its neighbors
    lbp = local_binary_pattern(edges, 8 * 1, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist / lbp_hist.sum()  

    # Combine features
    features = np.concatenate([hist.flatten(), fd, lbp_hist])
    return features


def augment_image(image):
    # Random rotation
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    # Random flipping
    if np.random.rand() > 0.5:
        rotated_image = cv2.flip(rotated_image, 1)

    return rotated_image

# Initialize features and labels
X = []
y = []

# Load images, augment, and extract features
for label, class_name in enumerate(class_names):
    class_folder = os.path.join(base_path, class_name)
    for filename in os.listdir(class_folder):
        image_path = os.path.join(class_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            # Resize to 256x256
            resized_image = cv2.resize(image, (256, 256))
            
            # Augment and extract features
            augmented_image = augment_image(resized_image)
            features = feature_extraction(augmented_image)
            X.append(features)
            y.append(class_name)

X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_resampled)
class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))

# Define KNN hyperparameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [20, 30, 40],
}

# Train KNN with RandomizedSearchCV
knn_classifier = KNeighborsClassifier()
random_search_knn = RandomizedSearchCV(knn_classifier, param_grid_knn, n_iter=50, cv=5, n_jobs=-1, verbose=1)
random_search_knn.fit(X_train, y_train)

# Get best model and evaluate
best_knn_classifier = random_search_knn.best_estimator_
print("Best Parameters for KNN:", random_search_knn.best_params_)
knn_pred = best_knn_classifier.predict(X_test)
print("KNN Performance:")
print(classification_report(y_test, knn_pred, target_names=label_encoder.classes_))

# Save model and label encoder
joblib.dump(best_knn_classifier, 'knn_pipeline.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
