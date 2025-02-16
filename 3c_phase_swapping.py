#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:39:18 2025

@author: cmalili
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Question 3c: Phase swapping
path_woman1 = "woman1.jpeg"
path_woman2 = "woman2.jpeg"

# Reading in image of woman1 and plotting it
def read_and_display(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    return img

woman1 = read_and_display(path_woman1)
woman2 = read_and_display(path_woman2)

# img is a single channel image
# function returns the magnitude spectrum and the phase spectru
def compute_dft_gray(img):        
# converting image of woman 1 to frequency domain and plotting magnitude and phase
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    magnitude, phase = cv2.cartToPolar(dft_shifted[:,:,0], dft_shifted[:,:,1])
    return magnitude, phase


r, g, b = cv2.split(woman1)

magnitude, phase = compute_dft_gray(r)
print(magnitude.shape)
magnitude = 1000*np.log(magnitude)
plt.imshow(magnitude)


def compute_dft_rgb(img):
    r, g, b = cv2.split(img)
    channels = [r, g, b]
    magnitudes = []
    phases = []
    for channel in channels:
        magnitude, phase = compute_dft_gray(channel)
        magnitudes.append(magnitude)
        phases.append(phase)
    return magnitudes, phases

woman1_magnitude, woman1_phase = compute_dft_rgb(woman1)
woman2_magnitude, woman2_phase = compute_dft_rgb(woman2)

'''
woman1_magnitude = 100*np.log(woman1_magnitude[0]).astype(np.uint8)
#woman1_magnitude = cv2.merge(woman1_magnitude.astype(np.uint8))
print(woman1_magnitude.shape)
plt.imshow(woman1_magnitude)
plt.show()

'''


def compute_idft_gray(magnitude, phase):
    real, imag = cv2.polarToCart(magnitude, phase)
    dft_shift = cv2.merge([real, imag])
    dft = np.fft.ifftshift(dft_shift)
    img = cv2.idft(dft)
    img = cv2.magnitude(img[:,:,0], img[:,:,1])
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    #img = np.clip(img, 0, 255).astype(np.uint8)
    img = (img).astype(np.uint8)
    return img

def compute_idft_rgb(magnitudes, phases):
    channels = []
    for magnitude, phase in zip(magnitudes, phases):
        channel = compute_idft_gray(magnitude, phase)
        img = compute_idft_gray(woman2_magnitude[2], woman2_phase[2])
        channels.append(channel)
    img = cv2.merge(channels)
    return img

img = compute_idft_rgb(woman2_magnitude, woman1_phase)
plt.imshow(img)
plt.show()

img = compute_idft_rgb(woman1_magnitude, woman2_phase)
plt.imshow(img)
plt.show()

