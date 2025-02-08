#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:15:15 2025

@author: cmalili
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Loading and displaying image in grayscale
path = "Boom-XB-1.jpg"
img = cv2.imread(path)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img)


# Computing the fourier transform of the image
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

#magnitude, phase = cv2.cartToPolar(dft_shift[], y)

magnitude, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])
magnitude_normalized = 2000*np.log(magnitude)
plt.imshow(magnitude_normalized)


phase_normalized = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_normalized)

'''
# Computing the low pass filter of the image
img_rows, img_cols = img.shape

mask = np.zeros((img_rows, img_cols, 2))
center_x, center_y = img_cols//2, img_rows//2
radius = 200

for row in range(img_rows):
    for col in range(img_cols):
        if np.sqrt((col - center_x)**2 + (row - center_y)**2) < radius:
            mask[row, col, :] = 1

plt.imshow(mask[:,:,1])

# Applying the filter to the shifted transform

dft_lpf = dft_shift*mask

magnitude_spectrum_lpf = cv2.magnitude(dft_lpf[:,:,0], dft_lpf[:,:,1])
magnitude_spectrum_lpt = 2000*np.log(magnitude_spectrum_lpf)
plt.imshow(magnitude_spectrum_lpt)


phase_spectrum_lpf = cv2.phase(dft_lpf[:,:,0], dft_lpf[:,:,1])
phase_spectrum_lpf = cv2.normalize(phase_spectrum_lpf, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_spectrum_lpf)

# reconstructing the image from the low pass filtered image
dft_lpf_ishift = np.fft.ifftshift(dft_lpf)
img_lpf = cv2.idft(dft_lpf_ishift)

img_lpf = cv2.magnitude(img_lpf[:,:,0], img_lpf[:,:,1])

plt.imshow(img_lpf)


# repeat procedure for both the
# high pass filter and 
# band pass filter


# Question 3c: 

'''













