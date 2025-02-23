#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:12:17 2025

@author: cmalili
"""

import cv2
import numpy as np

def generate_gaussian_pyramid(image, levels):
    gp = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        gp.append(image)
    return gp

def generate_laplacian_pyramid(image, levels):
    gp = generate_gaussian_pyramid(image, levels)
    lp = [gp[-1]]  # Start with the smallest image in the Gaussian pyramid
    for i in range(levels-1, -1, -1):
        size = (gp[i].shape[1], gp[i].shape[0])  # Width, Height
        laplacian = cv2.subtract(gp[i], cv2.pyrUp(gp[i+1], dstsize=size))
        lp.append(laplacian)
    return lp

def blend_pyramids(lpA, lpB, gpMask):
    blended_pyramid = []
    for la, lb, mask in zip(lpA, lpB, gpMask):
        mask = mask.astype(np.float32) / 255  # Normalize mask
        blended = la * mask + lb * (1.0 - mask)
        blended_pyramid.append(blended)
    return blended_pyramid

def reconstruct_from_pyramid(lp):
    image = lp[0]
    for i in range(1, len(lp)):
        size = (lp[i].shape[1], lp[i].shape[0])  # Width, Height
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, lp[i])
    return image
'''
# Load images and mask
image1 = cv2.imread('image1.jpg')  # Shape: (704, 800, 3)
image2 = cv2.imread('image2.jpg')  # Shape: (704, 800, 3)
mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)  # Shape: (704, 800)
'''
path_mango = "mango.jpg"
path_strawberry = "strawberry.png"

mango = cv2.imread(path_mango)
mango = cv2.cvtColor(mango, cv2.COLOR_BGR2RGB)
#mango = mango[:720,:800]
mango = mango[:704,:800]

strawberry = cv2.imread(path_strawberry)
strawberry = cv2.cvtColor(strawberry, cv2.COLOR_BGR2RGB)
#strawberry = strawberry[:,362:]
strawberry = strawberry[:704,362:]

# Creating a mango mask
x,y,_ = mango.shape
mask = np.zeros((x,y))
mask[:,:355] = 1

image1 = mango
image2 = strawberry

# Ensure all images are the same size
assert image1.shape == image2.shape == (704, 800, 3)
assert mask.shape == (704, 800)

# Set number of pyramid levels
levels = 5  

# Generate pyramids
lpA = generate_laplacian_pyramid(image1, levels)
lpB = generate_laplacian_pyramid(image2, levels)
gpMask = generate_gaussian_pyramid(mask, levels)

# Blend Laplacian pyramids
blended_pyramid = blend_pyramids(lpA, lpB, gpMask)

# Reconstruct the final blended image
blended_image = reconstruct_from_pyramid(blended_pyramid)

# Save and display
cv2.imwrite('blended_image.jpg', blended_image)
cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
