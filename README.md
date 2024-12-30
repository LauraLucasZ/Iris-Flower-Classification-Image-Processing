# Iris Flower Classification Project

## Overview
The Iris Flower Classification project uses image processing and machine learning techniques to classify images of iris flowers into three categories:
- **Iris-setosa**
- **Iris-versicolour**
- **Iris-virginica**

The project involves data preprocessing, feature extraction, model training, and GUI-based prediction. Two machine learning models, Random Forest and K-Nearest Neighbors (KNN), are implemented and evaluated for performance.

---

## Features
### 1. **Data Processing**
- **Segmentation**: Isolates the iris flower from the background using HSV masking.
- **Data Augmentation**: Includes random rotation and flipping for dataset enrichment.
- **Contrast Stretching**: Enhances image details by adjusting intensity levels.

### 2. **Feature Extraction**
- **Histogram of Oriented Gradients (HOG)**: Captures texture and shape.
- **Local Binary Pattern (LBP)**: Encodes local texture relationships.
- **Canny Edge Detection**: Highlights edges in the image.
- **Histogram Analysis**: Analyzes pixel intensity distributions.

### 3. **Handling Class Imbalance**
- Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to balance class distributions.

### 4. **Model Training**
- **Random Forest Classifier**: Hyperparameters optimized using RandomizedSearchCV.
- **K-Nearest Neighbors (KNN)**: Hyperparameters tuned with RandomizedSearchCV.

### 5. **GUI for Predictions**
A simple GUI built using Tkinter allows users to upload and classify iris flower images.

---

## Setup Instructions

### Prerequisites
Ensure the following are installed:
- Python (>= 3.8)
- OpenCV
- NumPy
- scikit-learn
- imbalanced-learn
- scikit-image
- joblib
- Tkinter
- PIL (Pillow)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/iris-flower-classification.git
   ```

2. Navigate to the project directory:
   ```bash
   cd iris-flower-classification
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset
Place the `IrisFlowerImages` folder containing subfolders `iris-setosa`, `iris-versicolour`, and `iris-virginica` in the specified directory:
```
/Users/yourName/Desktop/IrisFlowerImages
```

Each subfolder should contain the respective class images.
Dataset link: https://www.kaggle.com/datasets/jeffheaton/iris-computer-vision

---

## Running the Project

### 1. Train the Models
Run the training script to generate the Random Forest and KNN models:
For Random Forest:
```bash
python main.py
```
For k-NN:
```bash
python mainKnn.py
```

### 2. Launch the GUI
Use the GUI script for image classification:
For testing Random Forest:
```bash
python test.py
```
For testing k-NN:
```bash
python testKnn.py
```

---

## File Structure
```
.
├── IrisFlowerImages/           # Dataset folder
├── main.py                     # Script for training Random Forest
├── mainKnn.py                  # Script for training k-NN
├── test.py                     # GUI script for image classification using Random Forest
├── testKnn.py                  # GUI script for image classification using k-NN
├── rf_pipeline.pkl             # Trained Random Forest model
├── knn_pipeline.pkl            # Trained KNN model
├── label_encoder.pkl           # Label encoder
├── requirements.txt            # Python dependencies
```

---

## Model Performance
### Evaluation Metrics
- **Precision, Recall, F1-Score**: Analyzed using `classification_report` from scikit-learn.
- **Best Parameters**: Obtained via RandomizedSearchCV.

### Results
- Random Forest:
  - Best Parameters: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
 - Detailed performance metrics:

| Class               | Precision | Recall   | F1-Score | Support |
|---------------------|-----------|----------|----------|---------|
| iris-setosa         | 0.97      | 0.71     | 0.82     | 41      |
| iris-versicolour    | 0.61      | 0.96     | 0.75     | 26      |
| iris-virginica      | 0.90      | 0.80     | 0.85     | 35      |
| Accuracy            |           |          | 0.80     | 102     |
| Macro Average       | 0.83      | 0.82     | 0.80     | 102     |
| Weighted Average    | 0.85      | 0.80     | 0.81     | 102     |


- KNN:
  - Best Parameters: {'weights': 'distance', 'n_neighbors': 5, 'leaf_size': 20, 'algorithm': 'auto'}
 - Detailed performance metrics:

| Class               | Precision | Recall   | F1-Score | Support |
|---------------------|-----------|----------|----------|---------|
| iris-setosa         | 0.60      | 0.61     | 0.60     | 41      |
| iris-versicolour    | 0.42      | 0.38     | 0.40     | 26      |
| iris-virginica      | 0.44      | 0.46     | 0.45     | 35      |
| Accuracy            |           |          | 0.50     | 102     |
| Macro Average       | 0.49      | 0.48     | 0.48     | 102     |
| Weighted Average    | 0.50      | 0.50     | 0.50     | 102     |

---

## Future Improvements
- Support for more flower species.
- Use of deep learning models like CNNs for enhanced accuracy.
- Real-time image capture for classification.

---

## Credits
Developed by Laura as part of the **Image Processing Course Project**.

