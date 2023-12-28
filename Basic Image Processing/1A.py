import cv2
import numpy as np
from PIL import Image
img = cv2.imread("photo-1561731216-c3a4d99437d5.jpg",0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))
cv2.imshow('Image',res)
cv2.waitKey(0)
cv2.destroyWindow()
