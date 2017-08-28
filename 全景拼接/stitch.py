#coding=utf-8
from panorama import Stitcher
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
args = vars(ap.parse_args())

#加载两个图像并调整它们的宽度为400
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
#imageA = imutils.resize(imageA, width=400)
#imageB = imutils.resize(imageB, width=400)
 
# 初始化Stitcher()
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
 
cv2.imwrite('D:/images/vis.png',vis)
cv2.imwrite('D:/images/result.png',result)
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
