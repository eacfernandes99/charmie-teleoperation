import freenect
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import serial
import serial.tools.list_ports
import pyfirmata
import socket

localIP = "localhost"
localPort = 20000
bufferSize = 1024
adress = ("192.168.31.249", 20001)

UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDPServerSocket.setblocking(0)
UDPServerSocket.bind((localIP, localPort))

# ------------------------ Mediapipe ------------------------
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=0.5, circle_radius=0.5)
mp_holistic = mp.solutions.holistic # Holistic models
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ------------------ Functions definition -------------------

def calculate_angle(first, mid, end):

	first = np.array(first)
	mid = np.array(mid)
	end = np.array(end)

	mid2first = first - mid
	mid2end = end - mid

	cos_angle = np.dot(mid2first, mid2end) / (np.linalg.norm(mid2first) * np.linalg.norm(mid2end))
	joint_angle = np.arccos(cos_angle)

	return np.degrees(joint_angle)
	
def slope(p1, p2):
	x1, y1 = p1[0], p1[1]
	x2, y2 = p2[0], p2[1]
	try:
		m = (y2-y1)/(x2-x1)
		return m
	except ZeroDivisionError as e:
		m = (y2-y1)/(1)
		return m
		
def slope_deg(m1, m2):
	return np.rad2deg(np.arctan((m2-m1)/(1+m1*m2)))
	
def clamp(value, vmin, vmax):
	if value < vmin:
		out = vmin
	elif value > vmax:
		out = vmax
	else:
		out = value
	return out

def convert(x, in_min, in_Max, out_min, out_Max): 
    return max(min(out_Max, (x - in_min) * (out_Max - out_min) // (in_Max - in_min) + out_min), out_min)


open_pos_arm  = np.array([45, 150, 110, 125], dtype = float) # servo values at maximum angles
close_pos_arm = np.array([16,  40,  80,  55], dtype = float) # servo values at minimum angles

open_pos_hand =  [160, 15,   0,  20,  20] 
close_pos_hand = [80, 135, 110, 130, 120]

# ------------------------- Variables -------------------------	
L_shoulder_roll = 0
L_shoulder_pitch = 0
L_elbow_roll = 0
L_elbow_yaw = 0
L_wrist_yaw = 0
N = 5
avg_angle = []
joints = []
flag = 0

# ------------- Camera settings and configuration -------------

def get_video_kinect():
	array_video,_ = freenect.sync_get_video()
	array_video = cv2.cvtColor(array_video,cv2.COLOR_RGB2BGR)
	return array_video

def get_depth_kinect():
	array_depth,_ = freenect.sync_get_depth()
	return array_depth

def depth_to_distance(array, x, y):
	depth = 200 - 2.3*array[x,y] + 9.5e-3*(array[x,y]**2)
	return depth

device = cv2.CAP_OPENNI
capture = cv2.VideoCapture(device)

capture.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

capture.open(device)

while(True):
    start = time.time()

    color_image = get_video_kinect()

    # Depth calibration
    depth_image = get_depth_kinect()
    np.clip(depth_image, 0, 2**12-1, depth_image)
    depth_image >>= 2

    # Image recoloring - Convert the BGR image to RGB for Mediapipe
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = holistic.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        l_hand = results.left_hand_landmarks.landmark

        # 2D coordinates
        L_shoulder_2D = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x * 640,
                         landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y * 480]
        L_elbow_2D = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x * 640,
                      landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y * 480]
        L_wrist_2D = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x * 640,
                      landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y * 480]
        L_hip_2D = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x * 640,
                    landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y * 480]
        L_pinky_mcp_2D = [l_hand[mp_holistic.HandLandmark.PINKY_MCP.value].x * 640,
                          l_hand[mp_holistic.HandLandmark.PINKY_MCP.value].y * 480]
        L_middle_tip_2D = [l_hand[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x * 640,
                           l_hand[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y * 480]
        L_index_mcp_2D = [l_hand[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x * 640,
                          l_hand[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y * 480]
        
        if flag < 5:
            # Depth values for each joint
            print("Flag disabled, ", flag)
            flag = 0
            L_shoulder_depth = round(depth_to_distance(depth_image, int(L_shoulder_2D[0]), int(L_shoulder_2D[1])), 3)
            L_elbow_depth = round(depth_to_distance(depth_image, int(L_elbow_2D[0]), int(L_elbow_2D[1])), 3)
            L_wrist_depth = round(depth_to_distance(depth_image, int(L_wrist_2D[0]), int(L_wrist_2D[1])), 3)
            L_hip_depth = round(depth_to_distance(depth_image, int(L_hip_2D[0]), int(L_hip_2D[1])), 3)
            L_middle_tip_depth = round(depth_to_distance(depth_image, int(L_middle_tip_2D[0]),int(L_middle_tip_2D[1])), 3)
            depth_values = [L_shoulder_depth, L_elbow_depth, L_wrist_depth, L_shoulder_depth, L_middle_tip_depth]
            print(depth_values)
            for i in range(len(depth_values)):
                if 50 < depth_values[i] < 110:
                    flag = flag + 1
        if flag == 5:
            print("Flag enabled")
            cL_shoulder_depth = round(depth_to_distance(depth_image, int(L_shoulder_2D[0]), int(L_shoulder_2D[1])), 3)
            cL_elbow_depth = round(depth_to_distance(depth_image, int(L_elbow_2D[0]), int(L_elbow_2D[1])), 3)
            cL_wrist_depth = round(depth_to_distance(depth_image, int(L_wrist_2D[0]), int(L_wrist_2D[1])), 3)
            cL_hip_depth = round(depth_to_distance(depth_image, int(L_hip_2D[0]), int(L_hip_2D[1])), 3)
            cL_middle_tip_depth = round(depth_to_distance(depth_image, int(L_middle_tip_2D[0]),int(L_middle_tip_2D[1])), 3)
            c_depth_values = [cL_shoulder_depth, cL_elbow_depth, cL_wrist_depth, cL_shoulder_depth, cL_middle_tip_depth]
            for i in range(len(c_depth_values)):
                if abs(c_depth_values[i] - depth_values[i]) < 30:
                    depth_values[i] = c_depth_values[i]

			# 3D coordinates
            L_shoulder = [round(L_shoulder_2D[0], 3), round(L_shoulder_2D[1], 3), depth_values[0]]
            L_elbow = [round(L_elbow_2D[0], 3), round(L_elbow_2D[1], 3), depth_values[1]]
            L_wrist = [round(L_wrist_2D[0], 3), round(L_wrist_2D[1], 3), depth_values[2]]
            L_hip = [round(L_hip_2D[0], 3), round(L_hip_2D[1], 3), depth_values[3]]
            L_middle_tip = [round(L_middle_tip_2D[0], 3), round(L_middle_tip_2D[1], 3), depth_values[4]]

            # ------------------------------------- Angle calculation -------------------------------------
			
			# Left arm
            L_shoulder_roll = calculate_angle((L_hip[0], L_hip[1]),(L_shoulder[0], L_shoulder[1]),(L_elbow[0], L_elbow[1]))
            L_shoulder_roll = L_shoulder_roll * clamp((-(1/20)*(L_shoulder[2]-L_elbow[2])+1), 0, 1)
            L_shoulder_pitch = calculate_angle((L_hip[1], L_hip[2]),(L_shoulder[1], L_shoulder[2]),(L_elbow[1], L_elbow[2]))
            L_elbow_roll = calculate_angle(L_shoulder, L_elbow, L_wrist)
            L_elbow_yaw = calculate_angle((L_elbow[0]+20, L_elbow[2]), (L_elbow[0], L_elbow[2]), (L_wrist[0], L_wrist[2]))

            close_left = 0		
            if (np.sqrt((L_wrist[0]-L_middle_tip[0])**2+(L_wrist[1]-L_middle_tip[1])**2+(L_wrist[2]-L_middle_tip[2])**2)<45):
                close_left = 1

            L_wrist_yaw = 0
							
            if np.sum(avg_angle) == 0:
                L_shoulder_roll = [L_shoulder_roll] * N
                L_shoulder_pitch = [L_shoulder_pitch] * N
                L_elbow_roll = [L_elbow_roll] * N
                L_elbow_yaw = [L_elbow_yaw] * N
                L_wrist_yaw = [L_wrist_yaw] * N
                avg_angle = np.array([L_shoulder_roll, L_shoulder_pitch, L_elbow_roll, L_elbow_yaw, L_wrist_yaw]).reshape(5,5)

            else:
                avg_angle = np.delete(avg_angle, 0, 1)
                temp = np.array([L_shoulder_roll, L_shoulder_pitch, L_elbow_roll, L_elbow_yaw, L_wrist_yaw])
                avg_angle = np.column_stack((avg_angle, temp))
                avg_L_shoulder_roll = round((avg_angle[0,2]+2*avg_angle[0,3]+3*avg_angle[0,4])/6, 0)
                avg_L_shoulder_pitch = round((avg_angle[1,2]+2*avg_angle[1,3]+3*avg_angle[1,4])/6, 0)
                avg_L_elbow_roll = round((avg_angle[2,2]+2*avg_angle[2,3]+3*avg_angle[2,4])/6, 0)
                avg_L_elbow_yaw = round((avg_angle[3,2]+2*avg_angle[3,3]+3*avg_angle[3,4])/6, 0)
                avg_L_wrist_yaw = round((avg_angle[4,2]+2*avg_angle[4,3]+3*avg_angle[4,4])/6, 0)
			
            # Clamping values
			
            L_shoulder_roll = clamp(avg_L_shoulder_roll, 15, 85)
            L_shoulder_pitch = clamp(avg_L_shoulder_pitch, 0, 60)
            L_elbow_yaw = clamp(avg_L_elbow_yaw, 5, 30)
            L_elbow_roll = clamp(avg_L_elbow_roll, 25, 150)

            L_shoulder_roll_f = int(convert(L_shoulder_roll, 10, 90, close_pos_arm[0], open_pos_arm[0]))
            L_shoulder_pitch_f = int(convert(L_shoulder_pitch, 0, 65, close_pos_arm[1], open_pos_arm[1]))
            L_elbow_yaw_f = int(convert(L_elbow_yaw, 0, 35, close_pos_arm[2], open_pos_arm[2]))
            L_elbow_roll_f = int(convert(L_elbow_roll, 20, 155, close_pos_arm[3], open_pos_arm[3]))

            servo_values = [L_shoulder_roll_f, L_shoulder_pitch_f, L_elbow_yaw_f, L_elbow_roll_f]

            print(servo_values)

    except:
        print("Hand not detected")
        flag = 0

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
				   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
				   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
				  )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
				   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
				   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
				  )


    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

    cv2.imshow('Robot Arm', image)
    depth_show = depth_image.astype(np.uint8)
    cv2.imshow('Depth', depth_show)

    if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyAllWindows()
        board.exit()
        break
