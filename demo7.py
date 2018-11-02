# -- coding: utf-8 --
import cv2
import numpy as np

def main():
    '''
    GrabCut算法
    '''
    img = cv2.imread('demo7/img1.jpg')
    mask = np.zeros(img.shape[:2], np.uint8) # 0，表示指定为背景

    # 内部算法使用的数组，大小为 (1,65) 的 np.float64 类型零数组
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    rect = (375,235,70,250) # 该矩阵区域包含前景对象
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # mask现在包含0～3之间的值，将值为0、2的转为0，值为1、3的转为1
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

    grab = img*mask2[:,:,np.newaxis] # 使用mask2过滤值为0的像素，保留前景像素
    cv2.imwrite('demo7/grab.jpg', grab)


    '''
    分水岭算法
    '''
    img = cv2.imread('demo7/img2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 设置阈值，将图像中非白像素转化成黑色像素，并将黑白二值反转
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite('demo7/thresh.jpg', thresh)

    # 获取前景区域与背景区域
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations = 2) # 通过 morphologyEx 变换去除噪声数据
    sure_bg = cv2.dilate(opening,kernel,iterations = 3) # 通过对 morphologyEx 变换之后的图像进行膨胀操作，可以得到大部分都是背景的区域
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5) # 将远离背景区域的边界的点确定为前景
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(),255,0) # 应用阈值处理使获得确定的前景区域概率更高
    cv2.imwrite('demo7/sure_bg.jpg', sure_bg)
    cv2.imwrite('demo7/sure_fg.jpg', sure_fg)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg) # sure_bg与sure_fg可能存在重合，可从sure_bg与sure_fg的集合相减得到该不确定区域
    ret, markers = cv2.connectedComponents(sure_fg) # 设定“栅栏”阻止水汇聚

    # 在背景区域上加1，将unknown区域设为0
    markers = markers + 1
    markers[unknown==255] = 0

    # 最后打开门，让水漫起来并把栅栏绘成青色
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 255, 0]
    cv2.imwrite('demo7/water.jpg', img)


'''
图像分割
https://blog.csdn.net/eeeee123456/article/details/82968868
'''