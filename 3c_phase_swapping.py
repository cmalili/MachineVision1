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


