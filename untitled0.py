#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 21:23:51 2025

@author: cmalili
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt




.
# question 3d: Creating hybrid images

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

lpf_horse1 = low_pass_filter_rgb(horse1, 20)

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

hpf_horse2 = high_pass_filter_rgb(horse2, 20)

# Displaying low pass filtered rgb image
plt.imshow(hpf_horse2)
plt.show()

alpha = 0.5

horse12 = alpha*lpf_horse1 + (1-alpha)*hpf_horse2
horse12 = (horse12).astype(np.uint8)
plt.imshow(horse12)
plt.show()
