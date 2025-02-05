# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import cv2
#from PIL import Image

# Question 1a: Loading and displaying and image in opencv
path = "Estado_de_Mexico.jpg"
#image = Image.open()
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

# Question 1b: Cropping an image
crop = image[800:1000, 800:1000]
plt.imshow(crop)