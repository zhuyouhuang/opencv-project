#coding=utf-8
import numpy as np
import imutils
import cv2
 
class Stitcher:
	def __init__(self):
		#查询opencv版本
		self.isv3 = imutils.is_cv3()
	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# 加载图像，检测关键点，提取不变的描述符----SIFT
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
 
		#匹配两个图像之间的特征
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)
 
		
		if M is None:
			return None
		# M非空，则进行透视变换。
		#matches关键点匹配的列表，H为变换矩阵
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
 
		
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)
 
			return (result, vis)
		return result
		
	def detectAndDescribe(self, image):
		#检测关键点并提取局部不变描述符	
		# 转灰度
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		if self.isv3:
			#SIFT特征提取
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)
 
		# opencv2.X版本的
		else:
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)
 
		#关键点转换为数组
		kps = np.float32([kp.pt for kp in kps])		
		return (kps, features)
		
	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		#构建特征匹配器
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		# k-NN匹配
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
 
		for m in rawMatches:
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
			
		if len(matches) > 4:
			#构建两组点
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
 
			
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
			return (matches, H, status)
			
		return None
		
	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# 可视化两个图像之间的关键点
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
 
		
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			if s == 1:
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
 
		return vis
