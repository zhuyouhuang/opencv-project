#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pylab as plt

# 眼睛检测
def detectEyes(img):
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    faces = detectFaces(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = []
    for (x1, y1, x2, y2) in faces:
        roi_gray = gray[y1:y2, x1:x2]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 2)
        for (ex, ey, ew, eh) in eyes:
            result.append((x1+ex, y1+ey, x1+ex+ew, y1+ey+eh))
    return result


# 人脸检测
def detectFaces(img):
    face_cascade = cv2.CascadeClassifier(
        "haarcascades/haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        # 如果img维度为3，说明不是灰度图，先转化为灰度图gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x+width, y+height))
    return result


# 微笑检测
def detectSmiles(img):
    smiles_cascade = cv2.CascadeClassifier(
        "haarcascades/haarcascade_smile.xml")
    if img.ndim == 3:
        # 如果img维度为3，说明不是灰度图，先转化为灰度图gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    smiles = smiles_cascade.detectMultiScale(gray, 4, 5)
    result = []
    for (x, y, width, height) in smiles:
        result.append((x, y, x+width, y+height))
    return result


def process(img):
    '''在原图上框出头像，并显示'''
    faces = detectFaces(img)
    if faces:
        for (f1, f2, f3, f4) in faces:
            cv2.rectangle(img, (f1, f2), (f3, f4), (255, 255, 255),3)
    return img


if __name__ == '__main__':
    
    img=plt.imread('D:/nba.jpg')
    out=process(img)
    plt.imshow(out)
    plt.show()
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         # 水平翻转
#         frame = cv2.flip(frame, 1)
#         # 图像处理
#         image = process(frame)
#         cv2.imshow('window', image)
#         # 按下Esc 时退出
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
# 
#     cap.release()
#     cv2.destroyAllWindows()
