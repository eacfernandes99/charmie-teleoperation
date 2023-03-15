#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

bridge = CvBridge()
global sub_r

def right_eye_callback(msg_right):
	#print("Received an image right!")
	cv2_img_r = bridge.imgmsg_to_cv2(msg_right, desired_encoding='bgr8')
	img_r = np.fliplr(cv2_img_r)
	img_r = cv2.rotate(img_r, cv2.ROTATE_180)
	cv2.imshow("window_right", img_r)
	if cv2.waitKey(30) & 0xFF == 27:
		sub_r.unregister()
		cv2.destroyAllWindows()
				
if __name__ == '__main__':
	rospy.init_node('eye_sub', anonymous=False)
	sub_r = rospy.Subscriber("righteye", Image, right_eye_callback)
	rospy.spin()

	
