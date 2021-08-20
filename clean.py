from matplotlib import pyplot as plt
import cv2
img = cv2.imread("data/labels/00004N.png")
import numpy as np
print(np.unique(img))
plt.imshow(img)
plt.show()
