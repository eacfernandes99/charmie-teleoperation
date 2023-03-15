#!/usr/bin/env python
import roslib; roslib.load_manifest('beginner_tutorials')
import pyrealsense2 as rs
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import rospy
from std_msgs.msg import Float32MultiArray, Float32


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=0.5, circle_radius=0.5)
mp_holistic = mp.solutions.holistic # Holistic models
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the angles
   
def calculate_angle(first, mid, end):

	# Define the 3 points for angle calculation
	first = np.array(first)
	mid = np.array(mid)
	end = np.array(end)

	# Vector calculation
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
	
# ------------------------- Variables -------------------------	
N = 5
flag = 0
avg_angle = []
joints = []
img_w, img_h = 640, 480

# The camera matrix
focal_length = 1 * img_w

cam_matrix = np.array([ [focal_length, 0, img_h / 2],
		    	 [0, focal_length, img_w / 2],
		    	 [0, 0, 1]])

# The distortion parameters
dist_matrix = np.zeros((4, 1), dtype=np.float64)

# --------------------- Camera settings and configuration ---------------------

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Configuration of the resolution
config.enable_stream(rs.stream.depth, img_w, img_h, rs.format.z16, 30)
config.enable_stream(rs.stream.color, img_w, img_h, rs.format.bgr8, 30)
# Start streaming
profile = pipeline.start(config)
print("Connected to: ", profile.get_device())

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# Background removing
clipping_distance_in_meters = 1
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an aligned object
# rs.align allows a performed alignment of depth frames to others frames
# The "align_to" is the stream type to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# ------------------------- ROS nodes and topics -------------------------
rightarm_pub = rospy.Publisher('RightArm_Values', Float32MultiArray, queue_size=10)
leftarm_pub = rospy.Publisher('LeftArm_Values', Float32MultiArray, queue_size=10)
head_pub = rospy.Publisher('Head_Values', Float32MultiArray, queue_size=10)
rospy.init_node('NAO_Control', anonymous=False)



while True:
	start = time.time()

	# Wait for a coherent pair of frames: depth and color
	frames = pipeline.wait_for_frames()
	# Align the depth frame to color frame
	aligned_frames = align.process(frames)
	# Get aligned frames
	aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
	color_frame = aligned_frames.get_color_frame()
	# Validate that both frames are valid
	if not aligned_depth_frame or not color_frame:
		continue
	depth_image = np.asanyarray(aligned_depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())
	# Remove background - Set pixels further than clipping_distance to grey
	depth_image_3d = np.dstack(
	(depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
	bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 153, color_image)
	# Render images:
	#   depth align to color on left
	#   depth on right
	depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
	images = np.hstack((bg_removed, depth_colormap))

	# Image recoloring - Convert the BGR image to RGB for Mediapipe
	image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

	# To improve performance
	image.flags.writeable = False
	
	# Get the result
	results = holistic.process(image)
	results_face = face_mesh.process(image)
	
	# To improve performance
	image.flags.writeable = True

	# Convert the color space from RGB to BGR
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	face_3d = []
	face_2d = []
	
	try:
		landmarks = results.pose_landmarks.landmark
		r_hand = results.right_hand_landmarks.landmark
		l_hand = results.left_hand_landmarks.landmark
		
		if results_face.multi_face_landmarks:
			for face_landmarks in results_face.multi_face_landmarks:
				for idx, lm in enumerate(face_landmarks.landmark):			
					if idx == 104 or idx == 333 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
						if idx == 1:
							nose_2d = (lm.x * img_w, lm.y * img_h)
							nose_3d = (lm.x * img_w, lm.y * img_h, aligned_depth_frame.get_distance(int(lm.x * img_w), int(lm.y * img_h))*100)

						x, y = int(lm.x * img_w), int(lm.y * img_h)

						# Get the 2D Coordinates
						face_2d.append([x, y])

						# Get the 3D Coordinates
						face_3d.append([x, y, lm.z])       
		    
				# Convert it to the NumPy array
				face_2d = np.array(face_2d, dtype=np.float64)

				# Convert it to the NumPy array
				face_3d = np.array(face_3d, dtype=np.float64)

				# Solve PnP
				success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

				# Get rotational matrix
				rmat, jac = cv2.Rodrigues(rot_vec)

				# Get angles
				angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

				# Get the y rotation degree
				x = angles[0] * 360
				y = angles[1] * 360
				z = angles[2] * 360

				# Display the nose direction
				nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

				p1 = (int(nose_2d[0]), int(nose_2d[1]))
				p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))

				cv2.line(image, p1, p2, (255, 0, 0), 3)

		
		# Get 2D coordinates
		R_shoulder_2D = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x * img_w,
				  landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y * img_h]
		R_elbow_2D = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x * img_w,
			      landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y * img_h]
		R_wrist_2D = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x * img_w,
			      landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y * img_h]
		R_hip_2D = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x * img_w,
			    landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y * img_h]
		R_pinky_mcp_2D = [r_hand[mp_holistic.HandLandmark.PINKY_MCP.value].x * img_w,
				   r_hand[mp_holistic.HandLandmark.PINKY_MCP.value].y * img_h]
		R_middle_tip_2D = [r_hand[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x * img_w,
				   r_hand[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y * img_h]
		R_index_mcp_2D = [r_hand[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x * img_w,
				   r_hand[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y * img_h]
		L_shoulder_2D = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x * img_w,
				 landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y * img_h]
		L_elbow_2D = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x * img_w,
			      landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y * img_h]
		L_wrist_2D = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x * img_w,
			      landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y * img_h]
		L_hip_2D = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x * img_w,
			    landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y * img_h]
		L_pinky_mcp_2D = [l_hand[mp_holistic.HandLandmark.PINKY_MCP.value].x * img_w,
				   l_hand[mp_holistic.HandLandmark.PINKY_MCP.value].y * img_h]
		L_middle_tip_2D = [l_hand[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x * img_w,
				    l_hand[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y * img_h]
		L_index_mcp_2D = [l_hand[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x * img_w,
				   l_hand[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y * img_h]
		
		# Depth values for each joint
		if flag < 10:
			print("Flag disabled, ", flag)
			flag = 0
			R_shoulder_depth = round(aligned_depth_frame.get_distance(int(R_shoulder_2D[0]), int(R_shoulder_2D[1]))*100, 0)
			R_elbow_depth = round(aligned_depth_frame.get_distance(int(R_elbow_2D[0]), int(R_elbow_2D[1]))*100, 0)
			R_wrist_depth = round(aligned_depth_frame.get_distance(int(R_wrist_2D[0]), int(R_wrist_2D[1]))*100, 0)
			R_hip_depth = round(aligned_depth_frame.get_distance(int(R_hip_2D[0]), int(R_hip_2D[1]))*100, 0)
			R_middle_tip_depth = round(aligned_depth_frame.get_distance(int(R_middle_tip_2D[0]),int(R_middle_tip_2D[1]))*100, 0)
			L_shoulder_depth = round(aligned_depth_frame.get_distance(int(L_shoulder_2D[0]), int(L_shoulder_2D[1]))*100, 0)
			L_elbow_depth = round(aligned_depth_frame.get_distance(int(L_elbow_2D[0]), int(L_elbow_2D[1]))*100, 0)
			L_wrist_depth = round(aligned_depth_frame.get_distance(int(L_wrist_2D[0]), int(L_wrist_2D[1]))*100, 0)
			L_hip_depth = round(aligned_depth_frame.get_distance(int(L_hip_2D[0]), int(L_hip_2D[1]))*100, 0)
			L_middle_tip_depth = round(aligned_depth_frame.get_distance(int(L_middle_tip_2D[0]),int(L_middle_tip_2D[1]))*100, 0)
			depth_values = [R_shoulder_depth, R_elbow_depth, R_wrist_depth, R_hip_depth, R_middle_tip_depth, L_shoulder_depth, L_elbow_depth, L_wrist_depth, L_hip_depth, L_middle_tip_depth]
			for i in range(len(depth_values)):
				if 50 < depth_values[i] < 200:
					flag = flag + 1
		if flag == 10:
			print("Flag enabled")
			cR_shoulder_depth = round(aligned_depth_frame.get_distance(int(R_shoulder_2D[0]), int(R_shoulder_2D[1]))*100, 0)
			cR_elbow_depth = round(aligned_depth_frame.get_distance(int(R_elbow_2D[0]), int(R_elbow_2D[1]))*100, 0)
			cR_wrist_depth = round(aligned_depth_frame.get_distance(int(R_wrist_2D[0]), int(R_wrist_2D[1]))*100, 0)
			cR_hip_depth = round(aligned_depth_frame.get_distance(int(R_hip_2D[0]), int(R_hip_2D[1]))*100, 0)
			cR_middle_tip_depth = round(aligned_depth_frame.get_distance(int(R_middle_tip_2D[0]),int(R_middle_tip_2D[1]))*100, 0)
			cL_shoulder_depth = round(aligned_depth_frame.get_distance(int(L_shoulder_2D[0]), int(L_shoulder_2D[1]))*100, 0)
			cL_elbow_depth = round(aligned_depth_frame.get_distance(int(L_elbow_2D[0]), int(L_elbow_2D[1]))*100, 0)
			cL_wrist_depth = round(aligned_depth_frame.get_distance(int(L_wrist_2D[0]), int(L_wrist_2D[1]))*100, 0)
			cL_hip_depth = round(aligned_depth_frame.get_distance(int(L_hip_2D[0]), int(L_hip_2D[1]))*100, 0)
			cL_middle_tip_depth = round(aligned_depth_frame.get_distance(int(L_middle_tip_2D[0]),int(L_middle_tip_2D[1]))*100, 0)
			c_depth_values = [cR_shoulder_depth, cR_elbow_depth, cR_wrist_depth, cR_hip_depth, cR_middle_tip_depth, cL_shoulder_depth, cL_elbow_depth, cL_wrist_depth, cL_hip_depth, cL_middle_tip_depth]
			for i in range(len(c_depth_values)):
				if abs(c_depth_values[i] - depth_values[i]) < 30:
					depth_values[i] = c_depth_values[i]
			
			# 3D coordinates
			R_shoulder = [round(R_shoulder_2D[0], 3), round(R_shoulder_2D[1], 3), R_shoulder_depth]
			R_elbow = [round(R_elbow_2D[0], 3), round(R_elbow_2D[1], 3), R_elbow_depth]
			R_wrist = [round(R_wrist_2D[0], 3), round(R_wrist_2D[1], 3), R_wrist_depth]
			R_hip = [round(R_hip_2D[0], 3), round(R_hip_2D[1], 3), R_hip_depth]
			R_middle_tip = [round(R_middle_tip_2D[0], 3), round(R_middle_tip_2D[1], 3), R_middle_tip_depth]
			L_shoulder = [round(L_shoulder_2D[0], 3), round(L_shoulder_2D[1], 3), L_shoulder_depth]
			L_elbow = [round(L_elbow_2D[0], 3), round(L_elbow_2D[1], 3), L_elbow_depth]
			L_wrist = [round(L_wrist_2D[0], 3), round(L_wrist_2D[1], 3), L_wrist_depth]
			L_hip = [round(L_hip_2D[0], 3), round(L_hip_2D[1], 3), L_hip_depth]
			L_middle_tip = [round(L_middle_tip_2D[0], 3), round(L_middle_tip_2D[1], 3), L_middle_tip_depth]

			# ------------------------------------- Angle calculation -------------------------------------

			# Neck
			Head_yaw = (p2[0] - p1[0])/4
			Head_pitch = (p2[1] - p1[1])/4
			# -------------------- Right arm --------------------
			R_shoulder_roll = - (calculate_angle((R_hip[0], R_hip[1]),(R_shoulder[0], R_shoulder[1]),(R_elbow[0], R_elbow[1]))-20)
			R_shoulder_roll = R_shoulder_roll * clamp((-(1/20)*(R_shoulder[2]-R_elbow[2])+1), 0, 1)
			R_shoulder_pitch = - calculate_angle((R_hip[1], R_hip[2]),(R_shoulder[1], R_shoulder[2]),(R_elbow[1], R_elbow[2]))
			R_elbow_roll = - (calculate_angle(R_shoulder, R_elbow, R_wrist) - 180)
			# ---- Right Elbow Yaw ----
			if (R_elbow[0] < R_shoulder[0] and R_elbow[1] < R_shoulder[1]):
				if (R_wrist[0] < R_elbow[0] and R_wrist[1] < R_elbow[1]): 
					R_elbow_yaw = calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
				elif (R_wrist[0] > R_elbow[0] and R_wrist[1] < R_elbow[1]): 
					R_elbow_yaw = calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
				elif (R_wrist[0] < R_elbow[0] and R_wrist[1] > R_elbow[1]): 
					R_elbow_yaw = - calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
				else:
					R_elbow_yaw = - calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
			elif (R_elbow[0] < R_shoulder[0] and R_elbow[1] > R_shoulder[1]):
				if (R_wrist[0] < R_elbow[0] and R_wrist[1] < R_elbow[1]): 
					R_elbow_yaw = calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
				elif (R_wrist[0] > R_elbow[0] and R_wrist[1] < R_elbow[1]): 
					R_elbow_yaw = - calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
				elif (R_wrist[0] < R_elbow[0] and R_wrist[1] > R_elbow[1]): 
					R_elbow_yaw = calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
				else:
					R_elbow_yaw = - calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
			elif (R_elbow[0] > R_shoulder[0] and R_elbow[1] > R_shoulder[1]):
				if (R_wrist[0] > R_elbow[0] and R_wrist[1] < R_elbow[1]): 
					R_elbow_yaw = - calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
				elif (R_wrist[0] > R_elbow[0] and R_wrist[1] > R_elbow[1]):
					R_elbow_yaw = - calculate_angle((R_elbow[1], R_elbow[2]-20), (R_elbow[1], R_elbow[2]), (R_wrist[1], R_wrist[2]))
			# ---- Hand Open/Close ----
			close_right = 0
			if (np.sqrt((R_wrist[0]-R_middle_tip[0])**2+(R_wrist[1]-R_middle_tip[1])**2+(R_wrist[2]-R_middle_tip[2])**2)<45):
				close_right = 1
			# ---- Wrist Yaw ----
			m1 = slope(R_index_mcp_2D, R_wrist_2D)
			m2 = slope(R_wrist_2D, R_pinky_mcp_2D)
			R_wrist_yaw = slope_deg(m1,m2) * 3
			
			# Left arm
			L_shoulder_roll = calculate_angle((L_hip[0], L_hip[1]),(L_shoulder[0], L_shoulder[1]),(L_elbow[0], L_elbow[1]))-20
			L_shoulder_roll = L_shoulder_roll * clamp((-(1/20)*(L_shoulder[2]-L_elbow[2])+1), 0, 1)
			L_shoulder_pitch = calculate_angle((L_hip[1], L_hip[2]),(L_shoulder[1], L_shoulder[2]),(L_elbow[1], L_elbow[2]))
			L_elbow_roll = calculate_angle(L_shoulder, L_elbow, L_wrist) - 180
			if (L_elbow[0] > L_shoulder[0] and L_elbow[1] < L_shoulder[1]):
				if (L_wrist[0] < L_elbow[0] and L_wrist[1] < L_elbow[1]): 
					L_elbow_yaw = - calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
				elif (L_wrist[0] > L_elbow[0] and L_wrist[1] < L_elbow[1]): 
					L_elbow_yaw = - calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
				elif (L_wrist[0] < L_elbow[0] and L_wrist[1] > L_elbow[1]): 
					L_elbow_yaw = calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
				else:
					L_elbow_yaw = calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
			elif (L_elbow[0] < L_shoulder[0] and L_elbow[1] > L_shoulder[1]):
				if (L_wrist[0] < L_elbow[0] and L_wrist[1] < L_elbow[1]): 
					L_elbow_yaw = calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
				elif (L_wrist[0] < L_elbow[0] and L_wrist[1] > L_elbow[1]):
					L_elbow_yaw = calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
			elif (L_elbow[0] > L_shoulder[0] and L_elbow[1] > L_shoulder[1]):
				if (L_wrist[0] < L_elbow[0] and L_wrist[1] < L_elbow[1]): 
					L_elbow_yaw = calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
				elif (L_wrist[0] > L_elbow[0] and L_wrist[1] < L_elbow[1]): 
					L_elbow_yaw = - calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
				elif (L_wrist[0] < L_elbow[0] and L_wrist[1] > L_elbow[1]): 
					L_elbow_yaw = calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
				else:
					L_elbow_yaw = - calculate_angle((L_elbow[1], L_elbow[2]-20), (L_elbow[1], L_elbow[2]), (L_wrist[1], L_wrist[2]))
			close_left = 0		
			if (np.sqrt((L_wrist[0]-L_middle_tip[0])**2+(L_wrist[1]-L_middle_tip[1])**2+(L_wrist[2]-L_middle_tip[2])**2)<45):
				close_left = 1
			
			#R_wrist_yaw = 0
			L_wrist_yaw = 0
			#Head_yaw = 0
			#Head_pitch = 0
							
			if np.sum(avg_angle) == 0:
				R_shoulder_roll = [R_shoulder_roll] * N
				R_shoulder_pitch = [R_shoulder_pitch] * N
				R_elbow_roll = [R_elbow_roll] * N
				R_elbow_yaw = [R_elbow_yaw] * N
				R_wrist_yaw = [R_wrist_yaw] * N
				L_shoulder_roll = [L_shoulder_roll] * N
				L_shoulder_pitch = [L_shoulder_pitch] * N
				L_elbow_roll = [L_elbow_roll] * N
				L_elbow_yaw = [L_elbow_yaw] * N
				L_wrist_yaw = [L_wrist_yaw] * N
				Head_yaw = [Head_yaw] * N
				Head_pitch = [Head_pitch] * N
				avg_angle = np.array([R_shoulder_roll, R_shoulder_pitch, R_elbow_roll, R_elbow_yaw, R_wrist_yaw, L_shoulder_roll, L_shoulder_pitch, L_elbow_roll, L_elbow_yaw, L_wrist_yaw, 							      Head_yaw, Head_pitch]).reshape(12,5)

			else:
				avg_angle = np.delete(avg_angle, 0, 1)
				temp = np.array([R_shoulder_roll, R_shoulder_pitch, R_elbow_roll, R_elbow_yaw, R_wrist_yaw, L_shoulder_roll, L_shoulder_pitch, L_elbow_roll, L_elbow_yaw, L_wrist_yaw, Head_yaw, Head_pitch])
				avg_angle = np.column_stack((avg_angle, temp))
				avg_R_shoulder_roll = round((avg_angle[0,2]+2*avg_angle[0,3]+3*avg_angle[0,4])/6, 0)
				avg_R_shoulder_pitch = round((avg_angle[1,2]+2*avg_angle[1,3]+3*avg_angle[1,4])/6, 0)
				avg_R_elbow_roll = round((avg_angle[2,2]+2*avg_angle[2,3]+3*avg_angle[2,4])/6, 0)
				avg_R_elbow_yaw = round((avg_angle[3,2]+2*avg_angle[3,3]+3*avg_angle[3,4])/6, 0)
				avg_R_wrist_yaw = round((avg_angle[4,2]+2*avg_angle[4,3]+3*avg_angle[4,4])/6, 0)
				avg_L_shoulder_roll = round((avg_angle[5,2]+2*avg_angle[5,3]+3*avg_angle[5,4])/6, 0)
				avg_L_shoulder_pitch = round((avg_angle[6,2]+2*avg_angle[6,3]+3*avg_angle[6,4])/6, 0)
				avg_L_elbow_roll = round((avg_angle[7,2]+2*avg_angle[7,3]+3*avg_angle[7,4])/6, 0)
				avg_L_elbow_yaw = round((avg_angle[8,2]+2*avg_angle[8,3]+3*avg_angle[8,4])/6, 0)
				avg_L_wrist_yaw = round((avg_angle[9,2]+2*avg_angle[9,3]+3*avg_angle[9,4])/6, 0)
				avg_Head_yaw = round((avg_angle[10,2]+2*avg_angle[10,3]+3*avg_angle[10,4])/6, 0)
				avg_Head_pitch = round((avg_angle[11,2]+2*avg_angle[11,3]+3*avg_angle[11,4])/6, 0)
				
				#joints = [avg_R_shoulder_roll, avg_R_shoulder_pitch, avg_R_elbow_roll, avg_R_elbow_yaw, avg_R_wrist_yaw, close_right, avg_L_shoulder_roll, avg_L_shoulder_pitch, avg_L_elbow_roll, avg_L_elbow_yaw, avg_L_wrist_yaw, close_left, avg_Head_yaw, avg_Head_pitch]
				rightarm_joints = [avg_R_shoulder_roll, avg_R_shoulder_pitch, avg_R_elbow_roll, avg_R_elbow_yaw, avg_R_wrist_yaw, close_right]
				leftarm_joints = [avg_L_shoulder_roll, avg_L_shoulder_pitch, avg_L_elbow_roll, avg_L_elbow_yaw, avg_L_wrist_yaw, close_left]
				head_joints = [avg_Head_yaw, avg_Head_pitch]

				rightarm_msg = Float32MultiArray()
				rightarm_msg.data = rightarm_joints
				
				leftarm_msg = Float32MultiArray()
				leftarm_msg.data = leftarm_joints
				
				head_msg = Float32MultiArray()
				head_msg.data = head_joints

				rightarm_pub.publish(rightarm_msg)
				leftarm_pub.publish(leftarm_msg)
				head_pub.publish(head_msg)
		
	except:
		print("Pose not detected")
		flag = 0
		
		rightarm_msg = Float32MultiArray()
		rightarm_msg.data = [0, 0, 0, 0, 0, 0]
		
		leftarm_msg = Float32MultiArray()
		leftarm_msg.data = [0, 0, 0, 0, 0, 0]
		
		head_msg = Float32MultiArray()
		head_msg.data = [0, 0]

		rightarm_pub.publish(rightarm_msg)
		leftarm_pub.publish(leftarm_msg)
		head_pub.publish(head_msg)
		
		
	# Render detections
	
	mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
				   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
				   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
				  )

	mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
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
	
	#print(joints)

	cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)


	cv2.imshow('Head Pose Estimation', image)

	if cv2.waitKey(5) & 0xFF == 27:
		pipeline.stop()
		break




