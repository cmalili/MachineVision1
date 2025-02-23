#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 12:26:44 2025

@author: cmalili
"""

# Question 4: Multiresolution blending using Gaussian/Laplacian pyramids

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Question 4a: Loading images of a strawberry and mango
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
'''
plt.imshow(strawberry)
plt.axis("off")
plt.show()

plt.imshow(mango)
plt.axis("off")
plt.show()

'''
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
        print(level)
    
    return residuals, img_copy

mango_residuals, downsample = computeLaplacianPyramid(mango, 5)
'''
print(len(mango_residuals))
'''
for residual in mango_residuals:
    plt.imshow(residual)
    plt.axis("off")
    plt.show()
   
plt.imshow(downsample)
plt.axis("off")
plt.show()

# Reconstructing the image from the Gaussian pyramids

def reconstructImage(residuals, downsample):
    residuals.reverse()
    for residual in residuals:
        x, y,_ = downsample.shape
        upsample = cv2.resize(downsample, (y*2, x*2), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        upsample = cv2.GaussianBlur(upsample, kernel, sigma)
        downsample = upsample + residual
    return downsample

mango_reconstruction = reconstructImage(mango_residuals, downsample)
'''

plt.imshow(mango_reconstruction)
plt.axis("off")
plt.show()
'''
strawberry_copy = strawberry.copy()


# Question 4c : Creating mask for mango and strawberry

# Creating a mango mask
x,y,_ = mango.shape
mask = np.zeros(mango.shape)
mask[:,:355,:] = 1
'''
plt.imshow(mask)
plt.axis("off")
plt.show()
'''
'''   
# Question 4d : Directly blending the image of the mango and strawberry

direct_blend = (mask*mango + (1-mask)*strawberry).astype(np.uint8)
plt.imshow(direct_blend)
plt.axis("off")
plt.show()

print(strawberry.shape)


# Question 4e : Alpha blending the image of the mango and strawberry

mask_filtered = cv2.GaussianBlur(mask, (101,101), 35)
alpha_blend = (mask_filtered*mango + (1-mask_filtered)*strawberry).astype(np.uint8)

plt.imshow(alpha_blend)
plt.axis("off")
plt.show()

'''
# Question 4f: Blending the image of the mango and strawberry using multiresolution
# blending

# creating Gaussian pyramids for the mask the mango and strawberry

print(mask.shape)
def computeGaussianPyramid(mask, levels):
    #mask = cv2.GaussianBlur(mask, (45,45), 21)
    pyramid = [mask]
    for i in range(levels):
        mask = cv2.GaussianBlur(mask, kernel, sigma)
        x, y, _ = mask.shape
        mask = cv2.resize(mask, (y//2, x//2), fx=0.5, fy=0.5)
        pyramid.append(mask)
    return pyramid

mask_pyramid = computeGaussianPyramid(mask, 5)


for residual in mask_pyramid:
    plt.imshow(residual)
    #plt.axis("off")
    plt.show()
'''
'''


blended_residuals = []
mango_residuals, mango_downsample = computeLaplacianPyramid(mango, 5)
strawberry_residuals, strawberry_downsample = computeLaplacianPyramid(strawberry, 5)

mango_downsample = (mango_downsample*mask_pyramid[-1]).astype(np.uint8)
strawberry_downsample = (strawberry_downsample*(1 -mask_pyramid[-1])).astype(np.uint8)
blended_downsample = mango_downsample + strawberry_downsample

for mango, strawberry, mask in zip(mango_residuals, strawberry_residuals, mask_pyramid[:-1]):
    blended_residual = (mango*mask + strawberry*(1-mask)).astype(np.uint8)
    blended_residuals.append(blended_residual)
    

blended_image = reconstructImage(blended_residuals, blended_downsample)

plt.imshow(blended_image)





