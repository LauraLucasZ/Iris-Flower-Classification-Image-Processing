import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
import cv2
import numpy as np
from PIL import Image, ImageTk

def apply_edge_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Unable to load image.")
        return None, None

    # Resize the image to 256x256
    resized_image = cv2.resize(image, (256, 256))

    # Convertion to HSV for color filtering
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # Defining the range for green color
    lower_green = np.array([30, 25, 25])  # Lower bound of green
    upper_green = np.array([90, 255, 255])  # Upper bound of green

    # Creating a mask for green pixels
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Inverting the mask to keep non-green areas
    non_green_mask = cv2.bitwise_not(green_mask)

    # Apply the mask to the original image
    segmented_image = cv2.bitwise_and(resized_image, resized_image, mask=non_green_mask)

    # Convert the non-green image to grayscale
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Contrast stretching
    smin = 0
    smax = 255 
    cutoff_fraction = 0.03
    total_pixels = gray_image.size
    num_ignore = int(total_pixels * cutoff_fraction)

    hist, bin_edges = np.histogram(gray_image.flatten(), bins=256, range=[0,256])
    lower_cutoff = np.searchsorted(np.cumsum(hist), num_ignore)
    upper_cutoff = 255 - np.searchsorted(np.cumsum(hist[::-1]), num_ignore)
    img_clipped = np.clip(gray_image, lower_cutoff, upper_cutoff)
    r_min_clipped = np.min(img_clipped)
    r_max_clipped = np.max(img_clipped)
    stretched = ((img_clipped - r_min_clipped) * ((smax - smin) / (r_max_clipped - r_min_clipped)) + smin).astype(np.uint8)

    # Canny edge detection
    edges = cv2.Canny(stretched, 100, 200)

    return resized_image, edges

# GUI Setup
def select_image():

    file_path = filedialog.askopenfilename(
        title="Select an Image", 
        filetypes=[("Image Files", "*.png"), 
                  ("Image Files", "*.jpg"),
                  ("Image Files", "*.jpeg"),
                  ("Image Files", "*.bmp"),
                  ("Image Files", "*.tiff")]
    )

    if not file_path:
        messagebox.showerror("Error", "No image selected.")
        return None

    return file_path

def display_images(original_image, edges):
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

def process_image():
    # Selecting the image file
    image_path = select_image()
    if image_path:
        # Applying edge detection on the selected image
        original, edges = apply_edge_detection(image_path)
        if original is not None and edges is not None:
            display_images(original, edges)

# GUI
root = tk.Tk()
root.title("Edge Detection Tool")
root.geometry("800x400")

# Add a label to show instructions
label = Label(root, text="Select an Image to Apply Edge Detection", font=("Arial", 14))
label.pack(pady=20)

# Add a button to load and process the image
process_button = Button(root, text="Process Image", font=("Arial", 12), command=process_image)
process_button.pack(pady=20)

# Labels to display the original and edge-detected images
original_label = Label(root)
original_label.pack(side="left", padx=20)

edge_label = Label(root)
edge_label.pack(side="right", padx=20)

# Run the Tkinter event loop
root.mainloop()
