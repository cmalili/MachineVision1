#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 23:01:32 2025

@author: cmalili
"""

# Question 4: Multiresolution blending using Gaussian/Laplacian pyramids

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Question 4a: Loading images of a strawberry and mango
path_mango = "red_cup.webp"
path_strawberry = "blue_cup.webp"

mango = cv2.imread(path_mango)
mango = cv2.cvtColor(mango, cv2.COLOR_BGR2RGB)
'''
mango = mango[400:,450:2550]
mango = cv2.resize(mango, None, fx=0.5, fy=0.5)
mango = mango[:720,13:1037]
'''
strawberry = cv2.imread(path_strawberry)
strawberry = cv2.cvtColor(strawberry, cv2.COLOR_BGR2RGB)
#strawberry = strawberry[:720,:]

# Make sure both images have the same dimensions
print(f'Original soccerball shape: {strawberry.shape}')
print(f'Original basketball shape: {mango.shape}')

# Ensure both images have exactly the same shape
height = min(mango.shape[0], strawberry.shape[0])
width = min(mango.shape[1], strawberry.shape[1])
mango = mango[:height, :width]
strawberry = strawberry[:height, :width]

print(f'Adjusted soccerball shape: {strawberry.shape}')
print(f'Adjusted basketball shape: {mango.shape}')

plt.imshow(strawberry)
plt.axis("off")
plt.show()

plt.imshow(mango)
plt.axis("off")
plt.show()

# Question 4b: Creating Gaussian/Laplacian pyramids
# Gaussian pyramid for the mango

kernel = (41, 41)
sigma = 25
def computeLaplacianPyramid(image, levels):
    residuals = []
    
    img_copy = image.copy()
    
    for level in range(levels):
        
        blurred_image = cv2.GaussianBlur(img_copy, kernel, sigma)
        x, y, _ = blurred_image.shape
        downsample = cv2.resize(blurred_image, (y//2, x//2), fx=0.5, fy=0.5)
        upsample = cv2.resize(downsample, (y, x), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        upsample = cv2.GaussianBlur(upsample, kernel, sigma)
        residual = img_copy - upsample
        residuals.append(residual)
        img_copy = downsample
        print(f"Level {level} shape: {img_copy.shape}")
    
    return residuals, img_copy

# Define number of levels to use for both images
levels = 4

mango_residuals, mango_downsample = computeLaplacianPyramid(mango, levels)
print(f'Mango downsample shape: {mango_downsample.shape}')

# Reconstructing the image from the Gaussian pyramids

def reconstructImage(residuals, downsample):
    residuals_copy = residuals.copy()
    residuals_copy.reverse()
    reconstruction = downsample.copy()
    
    for residual in residuals_copy:
        x, y, _ = reconstruction.shape
        upsample = cv2.resize(reconstruction, (residual.shape[1], residual.shape[0]), interpolation=cv2.INTER_CUBIC)
        upsample = cv2.GaussianBlur(upsample, kernel, sigma)
        reconstruction = upsample + residual
    
    return reconstruction

mango_reconstruction = reconstructImage(mango_residuals, mango_downsample)

mango_diff = np.abs(mango - mango_reconstruction)
mango_diff = mango_diff.mean()
print(f"Mean mango difference: {mango_diff}")

# Question 4c : Creating mask for mango and strawberry

# Creating a mask with the same dimensions as the images
x, y, _ = mango.shape
mask = np.zeros((x, y, 3))
mask[:, :620, :] = 1

plt.imshow(mask)
plt.axis("off")
plt.title("Mask")
plt.show()

# Question 4f: Blending the image of the mango and strawberry using multiresolution blending

# Creating Gaussian pyramids for the mask
def computeGaussianPyramid(mask, levels):
    pyramid = [mask]
    img_copy = mask.copy()
    
    for i in range(levels):
        img_copy = cv2.GaussianBlur(img_copy, kernel, sigma)
        x, y, _ = img_copy.shape
        img_copy = cv2.resize(img_copy, (y//2, x//2), fx=0.5, fy=0.5)
        pyramid.append(img_copy)
        print(f"Mask level {i} shape: {img_copy.shape}")
    
    return pyramid

mask_pyramid = computeGaussianPyramid(mask, levels)

# Laplacian pyramid for strawberry with the same number of levels
strawberry_residuals, strawberry_downsample = computeLaplacianPyramid(strawberry, levels)
print(f'Strawberry downsample shape: {strawberry_downsample.shape}')

# Verify that the shapes match at the bottom level
print(f'Mango downsample shape: {mango_downsample.shape}')
print(f'Strawberry downsample shape: {strawberry_downsample.shape}')
print(f'Mask bottom level shape: {mask_pyramid[-1].shape}')

# Blend the bottom level
blended_downsample = (mango_downsample * mask_pyramid[-1] + 
                      strawberry_downsample * (1 - mask_pyramid[-1])).astype(np.uint8)

# Blend the residuals
blended_residuals = []
for i, (mango_res, strawberry_res) in enumerate(zip(mango_residuals, strawberry_residuals)):
    mask_level = mask_pyramid[i]
    # Verify shapes
    print(f"Level {i} - Mango: {mango_res.shape}, Strawberry: {strawberry_res.shape}, Mask: {mask_level.shape}")
    
    blended_residual = (mango_res * mask_level + 
                        strawberry_res * (1 - mask_level)).astype(np.uint8)
    blended_residuals.append(blended_residual)

# Reconstruct the blended image
blended_image = reconstructImage(blended_residuals, blended_downsample)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(mango)
plt.axis("off")
plt.title("Basketball")

plt.subplot(132)
plt.imshow(strawberry)
plt.axis("off")
plt.title("Soccer Ball")

plt.subplot(133)
plt.imshow(blended_image)
plt.axis("off")
plt.title("Blended Image")

plt.tight_layout()
plt.show()

# Also show direct blending and alpha blending for comparison
# Direct blending
direct_blend = (mask * mango + (1 - mask) * strawberry).astype(np.uint8)

# Alpha blending
mask_filtered = cv2.GaussianBlur(mask, (101, 101), 35)
alpha_blend = (mask_filtered * mango + (1 - mask_filtered) * strawberry).astype(np.uint8)

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(direct_blend)
plt.axis("off")
plt.title("Direct Blend")

plt.subplot(132)
plt.imshow(alpha_blend)
plt.axis("off")
plt.title("Alpha Blend")

plt.subplot(133)
plt.imshow(blended_image)
plt.axis("off")
plt.title("Multiresolution Blend")

plt.tight_layout()
plt.show()