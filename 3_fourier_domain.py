#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:15:15 2025

@author: cmalili
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
'''
# Question 3a: Displaying the magnitude and phase of an image in frequency domain
# Loading and displaying image in grayscale
path = "Boom-XB-1.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

plt.imshow(img)
plt.axis("off")
plt.show()

# Computing the fourier transform of the image
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

magnitude_normalized = 2000*np.log(magnitude)
phase_normalized = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)

plt.imshow(magnitude_normalized)
plt.axis("off")
plt.show()

plt.imshow(phase_normalized)
plt.axis("off")
plt.show()


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
magnitude_lpf_n = 2000*np.log(magnitude_lpf)

plt.imshow(magnitude_lpf_n)
plt.axis("off")
plt.show()


phase_lpf = cv2.normalize(phase_lpf, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_lpf)

# reconstructing the image from the low pass filtered image
dft_lpf_ishift = np.fft.ifftshift(dft_lpf)
img_lpf = cv2.idft(dft_lpf_ishift)
img_lpf = cv2.magnitude(img_lpf[:,:,0], img_lpf[:,:,1])

plt.imshow(img_lpf)
plt.axis("off")
plt.show()


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
magnitude_hpf_n = 2000*np.log(magnitude_hpf)

plt.imshow(magnitude_hpf_n)
plt.axis("off")
plt.show()


phase_hpf = cv2.normalize(phase_hpf, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_hpf)


# reconstructing the image from the low pass filtered image
dft_hpf_ishift = np.fft.ifftshift(dft_hpf)
img_hpf = cv2.idft(dft_hpf_ishift)
img_hpf = cv2.magnitude(img_hpf[:,:,0], img_hpf[:,:,1])

plt.imshow(img_hpf)
plt.axis("off")
plt.show()


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
magnitude_bpf_n = 2000*np.log(magnitude_bpf)

plt.imshow(magnitude_bpf_n)
plt.axis("off")
plt.show()


phase_bpf = cv2.normalize(phase_bpf, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(phase_bpf)


# reconstructing the image from the low pass filtered image
dft_bpf_ishift = np.fft.ifftshift(dft_bpf)
img_bpf = cv2.idft(dft_bpf_ishift)
img_bpf = cv2.magnitude(img_bpf[:,:,0], img_bpf[:,:,1])

plt.imshow(img_bpf)
plt.axis("off")
plt.show()


# Question 3c: Phase swapping
path_woman1 = "woman1.jpeg"
path_woman2 = "woman2.jpeg"

# Reading in image of woman1 and plotting it
img_woman1 = cv2.imread(path_woman1)
img_woman1 = cv2.cvtColor(img_woman1, cv2.COLOR_BGR2GRAY)
plt.imshow(img_woman1)
plt.axis("off")
plt.show()

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
plt.axis("off")
plt.show()


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
plt.axis("off")
plt.show()


# reconstructing image with magnitude from woman2 and phase from woman1
real_woman21, imag_woman21 = cv2.polarToCart(magnitude_woman2, phase_woman1)
dft_woman21 = cv2.merge([real_woman21, imag_woman21])
dft_woman21_ishift = np.fft.ifftshift(dft_woman21)
idft_woman21 = cv2.idft(dft_woman21_ishift)
img_woman21 = cv2.magnitude(idft_woman21[:,:,0], idft_woman21[:,:,1])
plt.imshow(img_woman21)
plt.axis("off")
plt.show()
'''

# Question 3d: Creating a hybrid image from image of horse1 at low frequency and 
# image of horse 2 at high frequency

# loading image of horse 1
path_horse1 = "horse1.jpg"
horse1 = cv2.imread(path_horse1)
horse1 = horse1[0:2300, 0:3500]
horse1 = cv2.cvtColor(horse1, cv2.COLOR_BGR2RGB)

# displaying image of horse 1
plt.imshow(horse1)
plt.axis("off")
plt.show()


def low_pass_filter_gray(image, radius):
    # Low pass filter of the image of horse1
    rows, cols = image.shape
    mask = np.zeros_like(image)
    center_x, center_y = cols//2, rows//2
    
    for row in range(rows):
        for col in range(cols):
            if np.sqrt((col - center_x)**2 + (row - center_y)**2) < radius:
                mask[row, col] = 1
                
    # displaying the low pass filter
    plt.imshow(mask)
    plt.show()
    
    # Tranforming image of horse1 to the frequency domain
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])
    
    # Displaying magnitude spectrum
    magnitude_norm = 1000*np.log(magnitude)
    plt.imshow(magnitude_norm)
    plt.show()
    
    # Applying the filter to the magnitude of shifted transform
    lpf_magnitude = magnitude*mask
    # Displaying low pass filtered magnitude spectrum
    lpf_magnitude_norm = magnitude_norm*mask
    plt.imshow(lpf_magnitude_norm)
    plt.show()
    
    # converting filtered image to spatial domain
    lpf_dft_x, lpf_dft_y = cv2.polarToCart(lpf_magnitude, phase)
    lpf_dft = cv2.merge([lpf_dft_x, lpf_dft_y])
    lpf_dft = np.fft.ifftshift(lpf_dft)
    lpf = cv2.idft(lpf_dft)
    lpf = cv2.magnitude(lpf[:,:,0], lpf[:,:,1])
    lpf = cv2.normalize(lpf, None, 0, 255, cv2.NORM_MINMAX)
    lpf = np.clip(lpf, 0, 255,).astype(np.uint8)
    
    return lpf, magnitude, phase, mask



def low_pass_filter_rgb(image, radius):
    [red, green, blue] = cv2.split(image)
    
    channels = [red, green, blue]
    lpf_channels = []
    
    for channel in channels:  
        lpf_channel,_,_,_ = low_pass_filter_gray(channel, radius)
        lpf_channels.append(lpf_channel)   
    lpf = cv2.merge(lpf_channels) 
    
    return lpf

lpf_horse1 = low_pass_filter_rgb(horse1, 30)

# Displaying low pass filtered rgb image
plt.imshow(lpf_horse1)
plt.show()


# loading image of horse 1
path_horse2 = "horse2.jpg"
horse2 = cv2.imread(path_horse2)
horse2 = horse2[0:2300, 0:3500]
horse2 = cv2.cvtColor(horse2, cv2.COLOR_BGR2RGB)

# displaying image of horse 1
plt.imshow(horse2)
plt.axis("off")
plt.show()


def high_pass_filter_gray(image, radius):
    # Low pass filter of the image of horse1
    rows, cols = image.shape
    mask = np.zeros_like(image)
    center_x, center_y = cols//2, rows//2
    
    for row in range(rows):
        for col in range(cols):
            if np.sqrt((col - center_x)**2 + (row - center_y)**2) > radius:
                mask[row, col] = 1
                
    # displaying the low pass filter
    plt.imshow(mask)
    plt.show()
    
    # Tranforming image of horse1 to the frequency domain
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])
    
    # Displaying magnitude spectrum
    magnitude_norm = 1000*np.log(magnitude)
    plt.imshow(magnitude_norm)
    plt.show()
    
    # Applying the filter to the magnitude of shifted transform
    hpf_magnitude = magnitude*mask
    # Displaying low pass filtered magnitude spectrum
    hpf_magnitude_norm = magnitude_norm*mask
    plt.imshow(hpf_magnitude_norm)
    plt.show()
    
    # converting filtered image to spatial domain
    hpf_dft_x, hpf_dft_y = cv2.polarToCart(hpf_magnitude, phase)
    hpf_dft = cv2.merge([hpf_dft_x, hpf_dft_y])
    hpf_dft = np.fft.ifftshift(hpf_dft)
    hpf = cv2.idft(hpf_dft)
    hpf = cv2.magnitude(hpf[:,:,0], hpf[:,:,1])
    hpf = cv2.normalize(hpf, None, 0, 255, cv2.NORM_MINMAX)
    hpf = np.clip(hpf, 0, 255,).astype(np.uint8)
    
    return hpf, magnitude, phase, mask



def high_pass_filter_rgb(image, radius):
    [red, green, blue] = cv2.split(image)
    
    channels = [red, green, blue]
    hpf_channels = []
    
    for channel in channels:  
        hpf_channel,_,_,_ = high_pass_filter_gray(channel, radius)
        hpf_channels.append(hpf_channel)   
    hpf = cv2.merge(hpf_channels) 
    
    return hpf

hpf_horse2 = high_pass_filter_rgb(horse2, 5)

# Displaying low pass filtered rgb image
plt.imshow(hpf_horse2)
plt.show()

alpha = 0.5

horse12 = alpha*lpf_horse1 + (1-alpha)*hpf_horse2
plt.imshow(horse12)
plt.show()
