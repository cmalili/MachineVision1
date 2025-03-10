#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 06:45:51 2025

@author: cmalili
"""

import rawpy
import cv2
import numpy as np
import matplotlib.pyplot as plt

#rawpy.libraw.set_maximum_supported_memory_mb  # Set to 8GB (adjust if needed)


#outdoor_path = "outside_scene.dng"
reference_path = "original_background.dng"
reference_raw = rawpy.imread(reference_path)
reference_rgb = reference_raw.postprocess()

plt.imshow(reference_rgb)
plt.axis('off')
plt.show()

oclusion_path = "ocluded_background.dng"
oclusion_raw = rawpy.imread(oclusion_path)
oclusion_rgb = oclusion_raw.postprocess()

plt.imshow(oclusion_rgb)
plt.axis('off')
plt.show()


diff = cv2.absdiff(reference_rgb, oclusion_rgb)#).astype(np.uint8)
diff = cv2.GaussianBlur(diff, (5,5), 0)
diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


plt.imshow(diff_normalized)
plt.axis('off')
plt.show()
