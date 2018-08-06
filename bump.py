import cv2
import math
import numpy as np
#img=cv2.imread("IMG_20180728_123311_HDR.jpg",0)
#IMG_20180728_123311_HDR    IMG_20180728_123311_HDR.jpg
img=cv2.imread("IMG_20180728_123223_HDR.jpg",0)
img=cv2.resize(img,(400,300))
print img.shape
img=img[int(0.2*img.shape[0]):-1]
img_gauss=cv2.GaussianBlur(img, (49,49),sigmaX=2,sigmaY=2)
img_median=cv2.medianBlur(img_gauss,31)
img_diff=img_gauss-img_median+127
img_thres=img_diff
for i in range(img_thres.shape[0]):
	for j in range(img_thres.shape[1]):
		if(img_thres[i,j]>150):
			img_thres[i,j]=255
		else:
			img_thres[i,j]=0

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img_diff,kernel,iterations = 2)
dil = cv2.dilate(erosion,kernel,iterations = 2)
cv2.imshow("image",dil)
cv2.waitKey(0)
