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
mango = mango[:720,:800]

strawberry = cv2.imread(path_strawberry)
strawberry = cv2.cvtColor(strawberry, cv2.COLOR_BGR2RGB)
strawberry = strawberry[:,362:]

plt.imshow(strawberry)
plt.axis("off")
plt.show()

plt.imshow(mango)
plt.axis("off")
plt.show()


# Question 4b: Creating Gaussian/Laplacian pyramids
# Gaussian pyramid for the mango

mango_residuals = []

mango_copy = mango.copy()
for i in range(4):
    mango_residual = mango_copy - cv2.GaussianBlur(mango_copy, (5,5), 2)
    mango_residuals.append(mango_residual)
    x, y, _ = mango_copy.shape
    mango_copy = cv2.resize(mango_copy, (y//2, x//2), fx=2, fy=2)
    

for mango_residual in mango_residuals:
    plt.imshow(mango_residual)
    plt.axis("off")
    plt.show()
    
plt.imshow(mango_copy)


# Reconstructing the image from the Gaussian pyramids

for i in range(4):
    x, y,_ = mango_copy.shape
    mango_copy = cv2.resize(mango_copy, (y*2, x*2), fx=2, fy=2)
    mango_copy = cv2.GaussianBlur(mango_copy, (5,5), 2)
    mango_copy = mango_copy + mango_residuals[3-i]
    
plt.imshow(mango_copy)
plt.axis("off")
plt.show()

strawberry_copy = strawberry.copy()


# Question 4c : Creating mask for mango and strawberry

# Creating a mango mask
x,y,_ = mango_copy.shape
mask = np.zeros(mango_copy.shape)
mask[:,:355] = 1

plt.imshow(mask)
plt.axis("off")
plt.show()
   
   
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


# Question 4f: Blending the image of the mango and strawberry using multiresolution
# blending

# creating Gaussian pyramids for the mask the mango and strawberry

blend_residuals = []

mango_copy1 = mango.copy()
strawberry_copy1 = strawberry.copy()
mask_copy1 = mask.copy()

for i in range(4):
    mango_residual = mango_copy1 - cv2.GaussianBlur(mango_copy1, (55,55), 25)
    strawberry_residual = strawberry_copy1 - cv2.GaussianBlur(strawberry_copy1, (55,55), 25)
    mask_residual = cv2.GaussianBlur(mask_copy1, (55,55), 25)
    blend_residual = (mask_residual*mango_residual + (1-mask_residual)*
                      strawberry_residual).astype(np.uint8)
    blend_residuals.append(blend_residual)
    
    mango_residuals.append(mango_residual)
    x, y, _ = mango_copy1.shape
    mango_copy1 = cv2.resize(mango_copy1, (y//2, x//2), fx=2, fy=2)
    strawberry_copy1 = cv2.resize(strawberry_copy1, (y//2, x//2), fx=2, fy=2)
    mask_copy1 = cv2.resize(mask_copy1, (y//2, x//2), fx=2, fy=2)

plt.imshow(blend_residuals[0])

# Reconstructing blended image from gaussian pyramid
image_blend = (mask_copy1*mango_copy1 + (1-mask_copy1)*strawberry_copy1).astype(np.uint8)
plt.imshow(image_blend)


for i in range(4):
    x, y,_ = image_blend.shape
    image_blend = cv2.resize(image_blend, (y*2, x*2), fx=2, fy=2)
    image_blend = cv2.GaussianBlur(image_blend, (55,55), 25)
    image_blend = image_blend + blend_residuals[3-i]

plt.imshow(image_blend)
plt.axis("off")
plt.show()






