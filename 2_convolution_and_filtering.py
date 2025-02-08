#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:25:58 2025

@author: cmalili
"""
# Question 2: Convolution as Matrix Multiplication and Edge Filtering

import matplotlib.pyplot as plt
import cv2
import datetime
import numpy as np


H = [[1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     
     
     
     
     
     
     
     ]

# Question 2a:
def conv2dmatrix(image, H):
    start_time = datetime.datetime.now()
    
    convolution = H @ image
    stop_time = datetime.datetime.now()
    latency = stop_time - start_time
    return 


    
    
    
# Question 2b:
    
# Question 2c:
    
# Question 2d:
    
# Question 2e: