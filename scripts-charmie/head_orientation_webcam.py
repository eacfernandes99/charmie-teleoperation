import cv2
import numpy as np

lower_y = np.array([87, 100, 100])
upper_y = np.array([96,255,255])
img_w, img_h = 640, 480

# --------------------- Camera settings and configuration ---------------------

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv,lower_y,upper_y)
	mask2 = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
	#fill any small holes
	closing = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
	#remove noise
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
	#dilate to merge adjacent blobs
	dilation = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

	contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area < 500:
			continue
		M = cv2.moments(cnt)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
		cv2.drawContours(frame, cnt, -1, (0,255,0), 3)
		#cv2.imshow("re", dilation)
	cv2.imshow("Head", hsv)
	
	if cv2.waitKey(1) & 0xFF == 27:
		cv2.destroyAllWindows()
		break
