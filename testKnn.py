import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import joblib

# Load the trained KNN model and label encoder
knn_classifier = joblib.load('knn_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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
    fd, _ = hog(edges, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    lbp = local_binary_pattern(edges, 8 * 1, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist / lbp_hist.sum()  # Normalize LBP histogram

    # Combine features
    features = np.concatenate([hist.flatten(), fd, lbp_hist])
    return features


# Preprocess and classify image
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Unable to load image.")
            return
        
        # Resize the image to 256x256
        resized_image = cv2.resize(image, (256, 256))
        
        # Extract features from the resized image
        features = feature_extraction(resized_image)
        label = knn_classifier.predict([features])
        predicted_class = label_encoder.inverse_transform(label)
        
        # Display the resized image and prediction result
        display_image(resized_image, predicted_class[0])

# Display image and prediction result
def display_image(image, predicted_class):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_pil = ImageTk.PhotoImage(image_pil)
    image_label.config(image=image_pil)
    image_label.image = image_pil
    result_label.config(text=f"Predicted Class: {predicted_class}", font=("Arial", 16, "bold"))

# GUI setup
root = tk.Tk()
root.title("Iris Flower Classifier")
root.geometry("500x500")

Label(root, text="Select an Image to Classify", font=("Arial", 14)).pack(pady=20)
Button(root, text="Classify Image", font=("Arial", 12), command=classify_image).pack(pady=20)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=10)

root.mainloop()
