# -- coding: utf-8 --
import cv2
import numpy as np

def main():
    img = cv2.imread('demo6/img.jpg', 0)

    # 使用Sobel算子
    x = cv2.Sobel(img,cv2.CV_16S,1,0) # 计算x方向
    y = cv2.Sobel(img,cv2.CV_16S,0,1) # 计算y方向
    sobelX = cv2.convertScaleAbs(x) # 转回uint8
    sobelY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(sobelX,0.5,sobelY,0.5,0) # 组合得到最终结果

    # 使用laplacian算子
    laplacian = cv2.Laplacian(img,cv2.CV_64F,ksize = 3)
    laplacian = cv2.convertScaleAbs(laplacian)

    # 使用Canny算子
    canny = cv2.Canny(img, 50, 120)

    # 图像轮廓
    ret, thresh = cv2.threshold(img,127,255,0) # 阈值处理，将图像转变为二值图
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    bg = np.zeros(img.shape, dtype = np.uint8)
    contours = cv2.drawContours(bg, contours, -1, (255,255,255), 1) #给轮廓填充颜色

    cv2.imwrite('demo6/sobelX.jpg', sobelX)
    cv2.imwrite('demo6/sobelY.jpg', sobelY)
    cv2.imwrite('demo6/sobel.jpg', sobel)
    cv2.imwrite('demo6/laplacian.jpg', laplacian)
    cv2.imwrite('demo6/canny.jpg', canny)
    cv2.imwrite('demo6/contours.jpg', contours)


'''
特征检测/提取
https://blog.csdn.net/eeeee123456/article/details/82966948
'''