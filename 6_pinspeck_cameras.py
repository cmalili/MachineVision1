#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 01:23:58 2025

@author: cmalili
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Question 6b: Loading living_room video and displaying reference
video_path = "living_room.MOV"
cap = cv2.VideoCapture(video_path)

ret, ref_frame = cap.read()
ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)

plt.imshow(ref)
plt.axis("off")
plt.show()

# Question 6c: Writing out the difference video
# Convert reference frame to grayscale
ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)

# Initialize batimg (required by problem)
batimg = None

# Create a VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('livingroom_output.avi', fourcc, 20.0, (ref_frame.shape[1], ref_frame.shape[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Difference image (Choose reference frame strategy)
    diff = cv2.absdiff(ref_gray, gray)  # Simple frame subtraction

    # Normalize and apply colormap for better visualization
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_colored = cv2.applyColorMap(diff_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # Update batimg (Problem requirement)
    batimg = diff_colored  # Example manipulation of batimg

    # Show results
    cv2.imshow('Difference Video', diff_colored)
    out.write(diff_colored)

    # Press 'q' to exit early
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()