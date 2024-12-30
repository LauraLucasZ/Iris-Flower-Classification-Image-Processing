import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt

# Feature extraction function
def feature_extraction(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Unable to load image.")
        return None, None, None

    # Resize the image to 256x256
    resized_image = cv2.resize(image, (256, 256))

    # Segmentation
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 25, 25])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)
    segmented_image = cv2.bitwise_and(resized_image, resized_image, mask=non_green_mask)

    # Grayscale conversion
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Contrast Stretching
    smin = 0
    smax = 255
    cutoff_fraction = 0.03
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
    fd, _ = hog(edges, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    lbp = local_binary_pattern(edges, 8 * 1, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist / lbp_hist.sum()

    # Combine features
    features = np.concatenate([fd, lbp_hist])
    return resized_image, edges, features

# GUI Setup
def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png"), ("Image Files", "*.jpg"), ("Image Files", "*.jpeg")]
    )
    if not file_path:
        messagebox.showerror("Error", "No image selected.")
        return None
    return file_path

def display_images(original_image, edges, features):
    # Convert the original image to PhotoImage for Tkinter
    original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    original_image_pil = ImageTk.PhotoImage(original_image_pil)

    # Convert the edge image to PhotoImage for Tkinter
    edges_image_pil = Image.fromarray(edges)
    edges_image_pil = ImageTk.PhotoImage(edges_image_pil)

    # Update the labels to show both images
    original_label.config(image=original_image_pil)
    original_label.image = original_image_pil
    original_label.pack(side="left", padx=20)

    edge_label.config(image=edges_image_pil)
    edge_label.image = edges_image_pil
    edge_label.pack(side="right", padx=20)

    # Plot and show the features
    plt.figure(figsize=(10, 5))
    plt.title("Features")
    plt.plot(features, label="Extracted Features")
    plt.legend()
    plt.show()

def process_image():
    # Selecting the image file
    image_path = select_image()
    if image_path:
        # Apply feature extraction on the selected image
        original, edges, features = feature_extraction(image_path)
        if original is not None and edges is not None:
            display_images(original, edges, features)

# GUI
root = tk.Tk()
root.title("Feature Extraction Tool")
root.geometry("800x400")

# Add a label to show instructions
label = Label(root, text="Select an Image to Extract Features", font=("Arial", 14))
label.pack(pady=20)

# Add a button to load and process the image
process_button = Button(root, text="Process Image", font=("Arial", 12), command=process_image)
process_button.pack(pady=20)

# Labels to display the original and processed images
original_label = Label(root)
original_label.pack(side="left", padx=20)

edge_label = Label(root)
edge_label.pack(side="right", padx=20)

# Run the Tkinter event loop
root.mainloop()
