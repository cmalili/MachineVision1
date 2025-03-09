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

# Create a VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('livingroom_output.avi', fourcc, 30.0, (ref_frame.shape[1], ref_frame.shape[0]))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.GaussianBlur(frame_rgb, (5,5), 0)
    
    # Difference image (Choose reference frame strategy)
    diff = cv2.absdiff(ref_frame, frame_rgb)  # Simple frame subtraction
    
    # Normalize 
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imshow('Difference Video', diff_normalized)
    out.write(diff_normalized)
    
    
    # Press 'q' to exit early
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()