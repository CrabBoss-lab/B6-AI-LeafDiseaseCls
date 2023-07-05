# -*- codeing = utf-8 -*-
# @Time :2023/5/15 19:45
# @Author :yujunyu
# @Site :
# @File :test.py
# @software: PyCharm
from PIL import Image
import cv2

img1 = cv2.imread('img.png')
img2 = cv2.imread('h.png')

img = img2 - img1

cv2.imshow('img', img)
cv2.waitKey(0)
