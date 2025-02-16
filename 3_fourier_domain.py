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
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")
plt.show()

# Computing the fourier transform of the image


img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

magnitude_normalized = 1000*np.log(magnitude)
phase_normalized = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)

plt.imshow(magnitude_normalized)
plt.axis("off")
plt.show()

plt.imshow(phase_normalized)
plt.axis("off")
plt.show()


# Question 3b: Low pass filter, high pass filter, band pass filter in the frequency
# domain

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

radius = 10
lpf = low_pass_filter_rgb(img, radius)

plt.imshow(lpf)
plt.show()

'''
img_gray = cv2.cvtColor(lpf, cv2.COLOR_RGB2GRAY)
dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

magnitude_normalized = 1000*np.log(magnitude)
phase_normalized = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)

plt.imshow(magnitude_normalized)
plt.axis("off")
plt.show()
'''

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

hpf = high_pass_filter_rgb(img, 10)

# Displaying low pass filtered rgb image
plt.imshow(hpf)
plt.show()



'''
img_gray = cv2.cvtColor(hpf, cv2.COLOR_RGB2GRAY)
dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

magnitude_normalized = 1000*np.log(magnitude)
phase_normalized = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)

plt.imshow(magnitude_normalized)
plt.axis("off")
plt.show()
'''





def band_pass_filter_gray(image, radius_min, radius_max):
    # Low pass filter of the image of horse1
    rows, cols = image.shape
    mask = np.zeros_like(image)
    center_x, center_y = cols//2, rows//2
    
    for row in range(rows):
        for col in range(cols):
            norm = np.sqrt((col - center_x)**2 + (row - center_y)**2)
            if norm > radius_min and norm < radius_max:
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



def band_pass_filter_rgb(image, radius_min, radius_max):
    [red, green, blue] = cv2.split(image)
    
    channels = [red, green, blue]
    bpf_channels = []
    
    for channel in channels:  
        bpf_channel,_,_,_ = band_pass_filter_gray(channel, radius_min, radius_max)
        bpf_channels.append(bpf_channel)   
    bpf = cv2.merge(bpf_channels) 
    
    return bpf

radius_min = 1
radius_max = 20
bpf = band_pass_filter_rgb(img, radius_min, radius_max)

# Displaying low pass filtered rgb image
plt.imshow(bpf)
plt.show()
