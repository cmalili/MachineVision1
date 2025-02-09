#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:15:15 2025

@author: cmalili
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Question 3a: Displaying the magnitude and phase of an image in frequency domain
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


# Question 3b: Low pass filter, high pass filter, band pass filter in the frequency
# domain

# Low pass filter of the image
img_rows, img_cols = img.shape

mask_lpf = np.zeros((img_rows, img_cols, 2))
center_x, center_y = img_cols//2, img_rows//2
radius = 100

for row in range(img_rows):
    for col in range(img_cols):
        if np.sqrt((col - center_x)**2 + (row - center_y)**2) < radius:
            mask_lpf[row, col, :] = 1

plt.imshow(mask_lpf[:,:,1])

# Applying the filter to the shifted transform

dft_lpf = dft_shift*mask_lpf

magnitude_lpf, phase_lpf = cv2.cartToPolar(dft_lpf[:,:,0], dft_lpf[:,:,1])
magnitude_lpf = 2000*np.log(magnitude_lpf)
plt.imshow(magnitude_lpf)


phase_lpf = cv2.normalize(phase_lpf, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_lpf)

# reconstructing the image from the low pass filtered image
dft_lpf_ishift = np.fft.ifftshift(dft_lpf)
img_lpf = cv2.idft(dft_lpf_ishift)
img_lpf = cv2.magnitude(img_lpf[:,:,0], img_lpf[:,:,1])
plt.imshow(img_lpf)



# repeat procedure for both the
# high pass filter and
mask_hpf = np.zeros((img_rows, img_cols, 2))
center_x, center_y = img_cols//2, img_rows//2
radius = 200

for row in range(img_rows):
    for col in range(img_cols):
        if np.sqrt((col - center_x)**2 + (row - center_y)**2) > radius:
            mask_hpf[row, col, :] = 1

plt.imshow(mask_hpf[:,:,1])

# Applying the filter to the shifted transform

dft_hpf = dft_shift*mask_hpf

magnitude_hpf, phase_hpf = cv2.cartToPolar(dft_hpf[:,:,0], dft_hpf[:,:,1])
magnitude_hpf = 2000*np.log(magnitude_hpf)
plt.imshow(magnitude_hpf)


phase_hpf = cv2.normalize(phase_hpf, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_hpf)


# reconstructing the image from the low pass filtered image
dft_hpf_ishift = np.fft.ifftshift(dft_hpf)
img_hpf = cv2.idft(dft_hpf_ishift)
img_hpf = cv2.magnitude(img_hpf[:,:,0], img_hpf[:,:,1])
plt.imshow(img_hpf)
 

# band pass filter
mask_bpf = np.zeros((img_rows, img_cols, 2))
center_x, center_y = img_cols//2, img_rows//2
radius_upper_limit = 200
radius_lower_limit = 80

for row in range(img_rows):
    for col in range(img_cols):
        length = np.sqrt((col - center_x)**2 + (row - center_y)**2)
        if length < radius_upper_limit and length > radius_lower_limit:
            mask_bpf[row, col, :] = 1

plt.imshow(mask_bpf[:,:,1])

# Applying the filter to the shifted transform

dft_bpf = dft_shift*mask_bpf

magnitude_bpf, phase_bpf = cv2.cartToPolar(dft_bpf[:,:,0], dft_bpf[:,:,1])
magnitude_bpf = 2000*np.log(magnitude_bpf)
plt.imshow(magnitude_bpf)


phase_bpf = cv2.normalize(phase_bpf, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_bpf)


# reconstructing the image from the low pass filtered image
dft_bpf_ishift = np.fft.ifftshift(dft_bpf)
img_bpf = cv2.idft(dft_bpf_ishift)
img_bpf = cv2.magnitude(img_bpf[:,:,0], img_bpf[:,:,1])
plt.imshow(img_bpf)



# Question 3c: Phase swapping
path_woman1 = "woman1.jpeg"
path_woman2 = "woman2.jpeg"

# Reading in image of woman1 and plotting it
img_woman1 = cv2.imread(path_woman1)
img_woman1 = cv2.cvtColor(img_woman1, cv2.COLOR_BGR2GRAY)
plt.imshow(img_woman1)

# converting image of woman 1 to frequency domain and plotting magnitude and phase
dft_woman1 = cv2.dft(np.float32(img_woman1), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_woman1_shifted = np.fft.fftshift(dft_woman1)
magnitude_woman1, phase_woman1 = cv2.cartToPolar(dft_woman1_shifted[:,:,0],
                                                 dft_woman1_shifted[:,:,1])
magnitude_woman1_normalized = 100*np.log(magnitude_woman1)
plt.imshow(magnitude_woman1_normalized)

phase_woman1_normalized = cv2.normalize(phase_woman1, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_woman1_normalized)


# Reading in image of woman2 and plotting it
img_woman2 = cv2.imread(path_woman2)
img_woman2 = cv2.cvtColor(img_woman2, cv2.COLOR_BGR2GRAY)
plt.imshow(img_woman2)

# converting image of woman 2 to frequency domain and plotting magnitude and phase
dft_woman2 = cv2.dft(np.float32(img_woman2), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_woman2_shifted = np.fft.fftshift(dft_woman2)
magnitude_woman2, phase_woman2 = cv2.cartToPolar(dft_woman2_shifted[:,:,0],
                                                 dft_woman2_shifted[:,:,1])
magnitude_woman2_normalized = 100*np.log(magnitude_woman2)
plt.imshow(magnitude_woman2_normalized)

phase_woman2_normalized = cv2.normalize(phase_woman2, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_woman2_normalized)


# reconstructing image with magnitude from woman1 and phase from woman2
real_woman12, imag_woman12 = cv2.polarToCart(magnitude_woman1, phase_woman2)
dft_woman12 = cv2.merge([real_woman12, imag_woman12])
dft_woman12_ishift = np.fft.ifftshift(dft_woman12)
idft_woman12 = cv2.idft(dft_woman12_ishift)
img_woman12 = cv2.magnitude(idft_woman12[:,:,0], idft_woman12[:,:,1])
plt.imshow(img_woman12)


# reconstructing image with magnitude from woman2 and phase from woman1
real_woman21, imag_woman21 = cv2.polarToCart(magnitude_woman2, phase_woman1)
dft_woman21 = cv2.merge([real_woman21, imag_woman21])
dft_woman21_ishift = np.fft.ifftshift(dft_woman21)
idft_woman21 = cv2.idft(dft_woman21_ishift)
img_woman21 = cv2.magnitude(idft_woman21[:,:,0], idft_woman21[:,:,1])
plt.imshow(img_woman21)


# Question 3d: Creating a hybrid image from image of horse1 at low frequency and 
# image of horse 2 at high frequency

# loading image of horse 1
path_horse1 = "horse1.jpg"
img_horse1 = cv2.imread(path_horse1)
img_horse1 = img_horse1[0:2300, 0:3500]
img_horse1 = cv2.cvtColor(img_horse1, cv2.COLOR_BGR2GRAY)
plt.imshow(img_horse1)

#path_horse2 = "horse2.jpg"



# Low pass filter of the image of horse1
img_horse1_rows, img_horse1_cols = img_horse1.shape

mask_horse1_lpf = np.zeros((img_horse1_rows, img_horse1_cols, 2))
center_horse1_x, center_horse1_y = img_horse1_cols//2, img_horse1_rows//2
radius_horse1 = 100

for row in range(img_horse1_rows):
    for col in range(img_horse1_cols):
        if np.sqrt((col - center_horse1_x)**2 + (row - center_horse1_y)**2) < radius_horse1:
            mask_horse1_lpf[row, col, :] = 1

plt.imshow(mask_horse1_lpf[:,:,1])

# Tranforming image of horse1 to the frequency domain
dft_horse1 = cv2.dft(np.float32(img_horse1), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_horse1_shift = np.fft.fftshift(dft_horse1)


# Applying the filter to the shifted transform
dft_horse1_lpf = dft_horse1_shift*mask_horse1_lpf

magnitude_lpf, phase_lpf = cv2.cartToPolar(dft_lpf[:,:,0], dft_lpf[:,:,1])
magnitude_lpf = 2000*np.log(magnitude_lpf)
plt.imshow(magnitude_lpf)


phase_lpf = cv2.normalize(phase_lpf, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_lpf)






'''
'''













