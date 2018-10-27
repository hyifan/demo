# -- coding: utf-8 --
import cv2
import numpy as np

def main():
	# 图像反转
	img1 = cv2.imread('demo3/img1.jpg', 0)
	new_img1 = np.ones(img1.shape) * 255 - img1
	cv2.imwrite('demo3/new_img1.jpg', new_img1)

	# 对数变换（c = 1，v = 10）
	img2 = cv2.imread('demo3/img2.jpg', 0)
	img2 = img2 / 255 # 归一化为范围[0,1]
	new_img2 = np.log10(1+10 * img2)
	new_img2 = new_img2 * 255 # 转换为范围[0,255]
	cv2.imwrite('demo3/new_img2.jpg', new_img2)

	# 伽马变换（c = 1，γ = 0.6）
	img3 = cv2.imread('demo3/img3.jpg', 0)
	img3 = img3 / 255 # 归一化为范围[0,1]
	new_img3 = np.power(img3, 0.6)
	new_img3 = new_img3 * 255 # 转换为范围[0,255]
	cv2.imwrite('demo3/new_img3.jpg', new_img3)

'''
实现基本图像变换
https://blog.csdn.net/eeeee123456/article/details/82914373
'''