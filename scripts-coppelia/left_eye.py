#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
#import os

# ------------------------------ Variables -------------------------------
bridge = CvBridge()
global sub_l

# ------------------------- Socket communication -------------------------



def left_eye_callback(msg_left):
	#print("Received an image left!")
	cv2_img_l = bridge.imgmsg_to_cv2(msg_left, desired_encoding='bgr8')
	img_l = np.fliplr(cv2_img_l)
	img_l = cv2.rotate(img_l, cv2.ROTATE_180)
	cv2.imshow("window_left", img_l)
	if cv2.waitKey(30) & 0xFF == 27:
		sub_l.unregister()
		cv2.destroyAllWindows()
				
if __name__ == '__main__':
	rospy.init_node('left_eye_sub', anonymous=False)
	sub_l = rospy.Subscriber("lefteye", Image, left_eye_callback)
	rospy.spin()

	
