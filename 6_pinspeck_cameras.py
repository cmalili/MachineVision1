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

# getting the average of the first 50 frames as reference

num_frames = 50
ref_frames = []
for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ref_frames.append(frame)
ref_frame = np.mean(ref_frames, axis=0).astype(np.uint8)

    
plt.imshow(ref_frame)
plt.axis("off")
plt.show()

# Question 6c: Writing out the difference video
# Convert reference frame to grayscale
#ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
'''
# Create a VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('livingroom_output.avi', fourcc, 20.0, (ref_frame.shape[1], ref_frame.shape[0]))
#out = cv2.VideoWriter('living_room_output.avi', fourcc, 20.0, (ref_frame.shape[1], ref_frame.shape[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #gray = cv2.GaussianBlur(gray, (25,25), 0)
    #gray = cv2.GaussianBlur(gray, (25,25), 0)
    
    # Difference image (Choose reference frame strategy)
    #diff = cv2.absdiff(ref_gray, gray)  # Simple frame subtraction
    #diff = cv2.absdiff(ref, frame)  # Simple frame subtraction
    '''
'''
    # Normalize and apply colormap for better visualization
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_colored = cv2.applyColorMap(diff_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    '''
    # Show results
    #cv2.imshow('Difference Video', diff_colored)
    #out.write(diff_colored)
'''
    cv2.imshow('Difference Video', frame)
    #out.write(diff)

    # Press 'q' to exit early
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
'''
# Release resources
cap.release()
#out.release()
cv2.destroyAllWindows()