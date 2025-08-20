# medical-image-analysis---image-processing
Medical image analysis uses advanced image processing techniques to extract valuable information from medical images (e.g., X-rays, MRIs, CT scans).

import cv2
import numpy as np
from skimage import filters
from matplotlib import pyplot as plt

# Load the medical image (replace 'medical_image.jpg' with your file)
image = cv2.imread('medical_image.jpg', cv2.IMREAD_COLOR)

# Convert the image to grayscale for easier processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred_image, 50, 150)

# Thresholding for binary segmentation (Example: Tumor detection)
thresh_val = filters.threshold_otsu(gray_image)
binary_image = gray_image > thresh_val

# Plotting the original, grayscale, and edge-detected images
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Grayscale image
plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

# Edge-detected image
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection (Canny)")
plt.axis('off')

plt.tight_layout()
plt.show()
