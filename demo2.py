# -- coding: utf-8 --
import cv2

# 灰色图片直方图均衡
def gray():
	img = cv2.imread('demo2/gray.jpg', 0) # 加载灰度图片
	equal_img = cv2.equalizeHist(img) # 直方图均衡
	cv2.imwrite('demo2/equal_gray.jpg', equal_img) # 保存图片

# 彩色图片直方图均衡
def rgb():
	img = cv2.imread('demo2/img.jpg')  # 加载BGR图片
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # BGR转HSV
	img_v = img_hsv[:, :, 2] # 获取V分量
	equal_img_v = cv2.equalizeHist(img_v) # 对V分量进行直方图均衡
	img_hsv[:, :, 2] = equal_img_v # 将均衡后的V分量赋值给原图
	equal_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) # HSV转BGR
	cv2.imwrite('demo2/equal_img.jpg',equal_img) # 保存图片


'''
实现直方图均衡
https://blog.csdn.net/eeeee123456/article/details/82911410
'''