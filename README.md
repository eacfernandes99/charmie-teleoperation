# charmie-teleoperation

This repository includes all the scripts used in CHARMIE teleoperation. The dissertation associated with these scripts is "Service Robot Remote Control System (VR Hradset + Pose Estimation)"


This repository has 4 folders, each one containing scripts for different situations.

📁 **coppelia-environmet**

  📎 _NAO_3D_
  
This file is a Coppelia file, containing the simulation environmet to test different scripts. Inside NAO robot, there is a LUA script responsible to receive the python script values and convert them to robot movement, as well as sending the different images from each eye. All this communication is made via ROS Noetic.
     
📁 **scripts-charmie**

  📎 _arm-direct-control_
  
This python script controls the arm using the Intel Realsense camera. In order to run this script, the computer must be plugged in directly to the Arduino board.
     
  📎 _intel_t_op_cnt_
  
This python script is used to control the CHARMIE through teleoperation, using the contour method for the head pose estimation and Intel Realsense camera. This script sends the values to CHARMIE PC through sockets.
     
  📎 _intel_t_op_pnp_
  
This python script is used to control the CHARMIE through teleoperation, using the solvepnp method for the head pose estimation and Intel Realsense camera. This script sends the values to CHARMIE PC through sockets.
     
  📎 _kinect_t_op_
  
This script is the same as intel_t_op_pnp, but instead of using an Intel Realsense camera, it uses a Kinect XBOX 360.

📁 **scripts-coppelia**

  📎 _coppelia_script_cnt_
  
This script is used to control the simulated NAO robot, using contour method and sending the values through ROS.

  📎 _coppelia_script_pnp_
  
This script is used to control the simulated NAO robot, using solvepnp method and sending the values through ROS.

  📎 _left_eye & right_eye_

These scripts are responsible to receive the images coming from each NAO's eyes and display them.

📁 **web-framework**

  📎 _charmie-vision_
  
This script must run in CHARMIE PC, receiving the frames from a webcam and sending them to a web framework.

  📎 _coppelia-vision_
  
This script is used to receive vision frames from the simulation and send them to a web framework.




