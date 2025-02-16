#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:25:58 2025

@author: cmalili
"""
# Question 2: Convolution as Matrix Multiplication and Edge Filtering

import matplotlib.pyplot as plt
import cv2
import datetime
import numpy as np


H = np.array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
       [-1.,  0.,  1., -1.,  0.,  1.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0., -1.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
       [-1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1.],
       [ 0., -1.,  0.,  0., -1.,  0.,  0., -1.,  0.],
       [ 0.,  0., -1.,  0.,  0., -1.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0., -1.,  0.,  1., -1.,  0.,  1.],
       [ 0.,  0.,  0.,  0., -1.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.]])
     

# Question 2b:
def conv2dmatrix(image, H):
    start_time = datetime.datetime.now()
    
    convolution = H @ image.flatten()
    stop_time = datetime.datetime.now()
    latency = stop_time - start_time
    convolution.reshape((5,5))
    return convolution, latency

    
# Question 2c:

h = np.array([[1, 0, -1,],
              [1, 0, -1,],
              [1, 0, -1,]])

print(f"kernel: {h}")  

img = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"image: {img}")

def compute_h_matrix(h, img):
    img_rows, img_cols = img.shape
    h_rows, h_cols = h.shape
    conv_rows, conv_cols = h_rows + img_rows - 1, h_cols + img_cols - 1
    #print(f"conv_rows, conv_cols: {conv_rows, conv_cols}")
    
    
    h_matrix = np.zeros((conv_rows*conv_cols, img_rows*img_cols))
    
    X = np.zeros((conv_cols, img_cols))
    X_stack = np.zeros((conv_cols, img_cols))
    
    for col in range(X.shape[1]):
            X_stack[col:h_cols + col, col] = h[0]
    '''
    print(f"h[0]: {h[0]}")
    print(f"X_stack: {X_stack}")
    '''
    for h_row in range(1, h_rows):
        for col in range(X.shape[1]):
            X[col:h_cols + col, col] = h[h_row]
        X_stack = np.vstack((X_stack, X))
    
    #print(f"X_stack: {X_stack}")
    
    for img_row in range(img_rows):
        h_matrix[conv_cols*img_row:X_stack.shape[0] + conv_cols*img_row,
                 img_cols*img_row:X_stack.shape[1] + img_cols*img_row] = X_stack
    
    #print(f"h_matrix: {h_matrix}")
    '''
    img_vector = img.reshape(img_cols*img_rows)
    print(f"image_vector: {img_vector}")
    '''
    return h_matrix

h_matrix = compute_h_matrix(h, img)
print(f"h_matrix: {h_matrix}")


def convolve(h, img):
    H = compute_h_matrix(h, img)
    img_vector = img.reshape((img.shape[0]*img.shape[1]))
    output_img = H @ img_vector
    output_img = output_img.reshape((h.shape[0]+ img.shape[0]-1, 
                                     h.shape[1]+ img.shape[1]-1))
    return output_img

output_img = convolve(h, img)
print(f"output: {output_img}")
print(f"output_shape: {output_img.shape}")


'''
def convolve(image, kernel):
    """Perform convolution using matrix multiplication."""
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    output = np.zeros_like(image)
    
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Question 2d:
'''

def gaussian_kernel(size, sigma=1):
    """Generate a Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def sobel_filters(image):
    """Apply Sobel filters to compute gradients."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Gx = convolve(image, Kx)
    Gy = convolve(image, Ky)
    
    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / magnitude.max() * 255
    direction = np.arctan2(Gy, Gx)
    
    return magnitude, direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """Thin edges by suppressing non-maximum values."""
    image_height, image_width = gradient_magnitude.shape
    output = np.zeros_like(gradient_magnitude)
    
    angle = gradient_direction * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, image_height-1):
        for j in range(1, image_width-1):
            q, r = 255, 255
            
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q, r = gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q, r = gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]
            elif 67.5 <= angle[i, j] < 112.5:
                q, r = gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q, r = gradient_magnitude[i+1, j+1], gradient_magnitude[i-1, j-1]
            
            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                output[i, j] = gradient_magnitude[i, j]
            else:
                output[i, j] = 0
    
    return output

def threshold(image, low_threshold, high_threshold):
    """Apply double thresholding."""
    strong = 255
    weak = 75
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    
    output = np.zeros_like(image)
    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak
    
    return output, strong, weak

def hysteresis(image, strong, weak):
    """Apply hysteresis to finalize edges."""
    image_height, image_width = image.shape
    
    for i in range(1, image_height-1):
        for j in range(1, image_width-1):
            if image[i, j] == weak:
                if strong in [image[i+1, j-1], image[i+1, j], image[i+1, j+1], image[i, j-1], image[i, j+1], image[i-1, j-1], image[i-1, j], image[i-1, j+1]]:
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    
    return image

def canny_edge_detector(image, low_threshold=1, high_threshold=20):
    """Full Canny Edge Detection implementation."""
    image = image.astype(np.float32) / 255.0
    smoothed = convolve(image, gaussian_kernel(5, sigma=1))
    magnitude, direction = sobel_filters(smoothed)
    suppressed = non_maximum_suppression(magnitude, direction)
    thresholded, strong, weak = threshold(suppressed, low_threshold, high_threshold)
    final_edges = hysteresis(thresholded, strong, weak)
    
    return final_edges.astype(np.uint8)

# Example usage:
image = cv2.imread('Lionel_Messi.jpg', cv2.IMREAD_GRAYSCALE)
edges = canny_edge_detector(image)

plt.imshow(image)
plt.axis("off")
plt.show()

plt.imshow(edges)
plt.axis("off")
plt.show()

cv_edges = cv2.Canny(image, 50, 100)

plt.imshow(cv_edges)
plt.axis("off")
plt.show()
'''
'''
# Question 2e: