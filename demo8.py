# -- coding: utf-8 --
import cv2
import numpy as np


'''
haar实现人脸检测
'''
def haar():
    img = cv2.imread('demo8/img1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''
    人脸检测
    '''
    face_cascade = cv2.CascadeClassifier('demo8/cascades/haarcascade_frontalface_default.xml') # CascadeClassifier对象
    faces = face_cascade.detectMultiScale(gray, 1.3, 2) # 传递参数是 scaleFactor 和 minNeighbors，它们分别表示人脸检测过程中每次迭代时图像的压缩率以及每个人脸矩形保留近邻数目的最小值。返回值为人脸矩形数组，数组元素形式为 (x,y,w,h)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # 绘制蓝色矩形框

    '''
    眼睛检测
    '''
    left_eye_cascade = cv2.CascadeClassifier('demo8/cascades/haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier('demo8/cascades/haarcascade_righteye_2splits.xml')
    left_eyes = left_eye_cascade.detectMultiScale(gray, 2, 1)
    right_eyes = right_eye_cascade.detectMultiScale(gray, 2, 1)
    for (x,y,w,h) in left_eyes:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2) # 绘制黄色矩形框

    for (x,y,w,h) in right_eyes:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) # 绘制青色矩形框

    '''
    嘴巴检测
    '''
    smile_cascade = cv2.CascadeClassifier('demo8/cascades/haarcascade_smile.xml')
    smiles = smile_cascade.detectMultiScale(gray, 1.3, 9)
    for (x,y,w,h) in smiles:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) # 绘制红色矩形框

    cv2.imwrite('demo8/rectangle.jpg', img)


'''
HOG实现行人检测
'''
# o, i 为矩形，形式为 (x,y,w,h)
# 该函数用于检测 矩形i 是否完全包含 矩形o，是返回true
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy +ih


# 该函数用于绘制最后检测到的矩形
def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 255), 2)


def hog():
    img = cv2.imread('demo8/img2.jpg')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # 返回值found为行人矩形数组，数组元素形式为 (x,y,w,h)
    found, w = hog.detectMultiScale(img)

    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                # 若is_inside(r, q)为TRUE，表示r完全包含在q中，这说明检测出现了错误
                break
            else:
                found_filtered.append(r)

    for person in found_filtered:
        draw_person(img, person)

    cv2.imwrite('demo8/people.jpg', img)


'''
haar实现人脸检测；HOG实现行人检测
https://blog.csdn.net/eeeee123456/article/details/82968988
'''