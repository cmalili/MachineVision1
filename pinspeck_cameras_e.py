#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 06:45:51 2025

@author: cmalili
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

reference_path = "reference.jpg"
oclusion_path = "oclusion.jpg"

reference = cv2.imread(reference_path)
reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

oclusion = cv2.imread(oclusion_path)
oclusion = cv2.cvtColor(oclusion, cv2.COLOR_BGR2RGB)

diff = (np.absdiff(reference, oclusion)).astype(np.uint8)
diff = cv2.GaussianBlur(diff, (5,5), 0)
diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

plt.imshow(reference)
plt.axis('off')
plt.show()

plt.imshow(oclusion)
plt.axis('off')
plt.show()

plt.imshow(diff_normalized)
plt.axis('off')
plt.show()
