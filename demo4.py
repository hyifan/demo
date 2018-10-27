# -- coding: utf-8 --
import cv2
import numpy as np

def main():
	img = cv2.imread('demo4/椒盐噪声.jpg')

	# 算术均值滤波器
	img1 = cv2.blur(img, (3,3))
	# 谐波均值滤波器
	img2 = 1/cv2.blur(1/(img+1E-10), (3,3)) # 加1E-10防止除0操作
	# 中值滤波器
	img3 = cv2.medianBlur(img, 3)

	cv2.imwrite('demo4/img1.jpg',img1)
	cv2.imwrite('demo4/img2.jpg',img2)
	cv2.imwrite('demo4/img3.jpg',img3)

	# Laplacian算子锐化图像
	kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
	img = cv2.imread('demo4/原图.jpg', 0)
	edge_img = cv2.filter2D(img, -1, kernel)
	output_img = cv2.add(img, edge_img)

	cv2.imwrite('demo4/edge_img.jpg', edge_img)
	cv2.imwrite('demo4/output_img.jpg', output_img)


'''
实现基本空间滤波器
https://blog.csdn.net/eeeee123456/article/details/82934582
'''