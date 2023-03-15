#!/usr/bin/env python
from flask import Flask,render_template,Response
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

app=Flask(__name__,template_folder='template')

bridge = CvBridge()

cv_image = 0
cv_image2 = 0


def callback(data):
    global cv_image
    cv2_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    cv_image_flip = np.fliplr(cv2_img)
    cv_image = cv2.rotate(cv_image_flip, cv2.ROTATE_180)
 
def callback2(data2):
    global cv_image2
    cv2_img2 = bridge.imgmsg_to_cv2(data2, desired_encoding='bgr8')
    cv_image_flip2 = np.fliplr(cv2_img2)
    cv_image2 = cv2.rotate(cv_image_flip2, cv2.ROTATE_180)
      
rospy.init_node('Robot_Vision', anonymous=False)
sub_l = rospy.Subscriber("LeftEye_Image", Image, callback)
sub_r = rospy.Subscriber("RightEye_Image", Image, callback2)


def generate_frames():
    global cv_image, cv_image2
    while True: 
        img = np.concatenate([cv_image, cv_image2], axis = 1)	
        ret,buffer=cv2.imencode('.jpg', img)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
