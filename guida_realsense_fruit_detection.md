# 1 CAMERA REALSENSE NODE
- roslaunch realsense2_camera rs_camera.launch align_depth:=true
- roslaunch realsense2_camera rs_aligned_depth.launch 

# 2 CAMERA REALSENSE + SLAM 
roslaunch rtabmap_ros rtabmap.launch \
    rtabmap_args:="--delete_db_on_start --Optimizer/GravitySigma 0.3" \
    depth_topic:=/camera/aligned_depth_to_color/image_raw \
    rgb_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info \
    approx_sync:=false \
    wait_imu_to_init:=false \
    imu_topic:=/rtabmap/imu \
    rviz:=true
    
# 3 OBJDET
roslaunch darknet_ros darknet_ros_rs.launch

# 4 Macchina a stati con plant detector
rosrun tesi_gripper plant_detector_node.py 

# 5 Raspberry images:
- ssh fabio@192.168.137.204
- pw: raspberry
- export ROS_HOSTNAME=fabio-pc
- export ROS_MASTER_URI=http://James:11311

# 6 Force feedback KNN
rosrun tesi_gripper force_estimation_node.py 

# ATI Sensor:
- ATI: collegare PoE nella parte sotto del PC e alla presa sotto
ref: http://wiki.ros.org/netft_utils
- roscore su james con pw altair
- rosrun netft_utils netft_utils_sim
- rostopic echo /netft_data


# Per visualizzare trasformazioni:
- tf_echo <source_frame> <target_frame> 
- rosrun tf tf_echo /panda_link0 /panda_link8
- rosrun tf tf_echo /panda_link0 /panda_EE

- Translation: [0.106, 0.136, 0.717]
- Rotation: in Quaternion [0.638, -0.293, 0.669, -0.245]
            in RPY (radian) [-1.548, -0.789, -1.666]
            in RPY (degree) [-88.694, -45.229, -95.459]


# Per eseguire traiettorie:
1) lanciare force control sul panda (ACTIVATE FCI from 192.168.3.1/desk!!!): 
- ssh koga@Koga 
- pw: solita
- roslaunch panda_force_ctrl panda_force_ctrl.launch 

2) lanciare nodo dal mio pc
roslaunch panda_controller PD_controller.launch 

3) ack per far partire la traiettoria pubblicata
rqt -> plugins -> services -> service caller -> toogle play -> call


# INSTALL PANDA LIBRARY
- https://projects.saifsidhik.page/panda_robot/
- https://github.com/justagist/panda_robot
- sudo pip install panda-robot
- sudo pip install numba

- sudo apt install ros-noetic-libfranka
- sudo apt install ros-noetic-franka-ros
- sudo apt install ros-noetic-panda-moveit-config

- sudo pip install rospy_message_converter
- rosdep install --from-paths src --ignore-src -r -y


# HAND EYE:
- roslaunch realsense2_camera rs_camera.launch align_depth:=true
- roslaunch easy_handeye calibrate_agri.launch
- take sample; export uri

# RESULTS (camera al contrario):
translation: 
  x: -0.06413974008111703
  y: 0.01353977082165731
  z: 0.008099189771108238
rotation: 
  x: 0.0063076722723766155
  y: 0.0030729916767021637
  z: 0.38461548288030684
  w: 0.9230502154928559
  
# RESULTS (camera giusta 16-03-23)
translation: 
  x: -0.018134767575568995
  y: 0.06503792173581778
  z: 0.005427418426183016
rotation: 
  x: 0.00856241157001291
  y: -0.0010321181654774358
  z: -0.9148889258318601
  w: 0.40361351963285763
  
# RESULTS con panda_link0 entrambi
translation: 
  x: -0.0174412922138256
  y: 0.06954448984302664
  z: 0.03354067298595653
rotation: 
  x: 0.006822312424831188
  y: -0.0020575854544159222
  z: -0.9210254510430451
  w: 0.38943721050540436
  







