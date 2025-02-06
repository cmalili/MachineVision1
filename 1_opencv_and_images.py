# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import cv2
#from PIL import Image

# Question 1a: Loading and displaying and image in opencv
path = "Estado_de_Mexico.jpg"
#image = Image.open()
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

# Question 1b: Cropping an image
crop = image[800:1000, 800:1000]
plt.imshow(crop)

# Question 1c: Downsampling a new image by a factor of 10
path1 = "Augmented_Instructions.jpg"
image1 = cv2.imread(path1)
scale_factor = 10
(h,w) = image1.shape[:2]

print(h)
print(w)

scaled_down_h, scaled_down_w = h//10, w//10

image_downsample = cv2.resize(image1, (scaled_down_w, scaled_down_h), fx=scale_factor, fy=scale_factor)
plt.imshow(image_downsample)


# Question 1d: Upsampling the downsampled image by a factor of 10
# Upsampling using nearest neighbor interpolation method
image_upsample_nearest = cv2.resize(image_downsample, (w, h), fx=scale_factor, fy=scale_factor,
                            interpolation=cv2.INTER_NEAREST)
plt.imshow(image_upsample_nearest)


# Upsampling using bicubic interpolation method
image_upsample_cubic = cv2.resize(image_downsample, (w, h), fx=scale_factor, fy=scale_factor,
                            interpolation=cv2.INTER_CUBIC)
plt.imshow(image_upsample_cubic)

# Question 1e: absolute difference between original image and upscaled sampled image

# absolute difference between original image and image upscaled with nearest neighbor
# interpolation
diff_nearest = cv2.absdiff(image1, image_upsample_nearest)
plt.imshow(diff_nearest)
sum_diff_nearest = cv2.sumElems(diff_nearest)
print(sum_diff_nearest)

# absolute difference between original image and image upscaled with cubic interpolation
diff_cubic = cv2.absdiff(image1, image_upsample_cubic)
plt.imshow(diff_cubic)
sum_diff_cubic = cv2.sumElems(diff_cubic)
print(sum_diff_cubic)
