#!/usr/bin/env python

# ROS libraries
import rospy
from darknet_ros_msgs.msg import BoundingBoxes
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import tf
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Quaternion, Point


# Python libraries
import pyrealsense2 as rs
from pynput import keyboard
import numpy as np
import cv2
from copy import copy
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import shutil
from beepy import beep
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from yolov3_tf2.models import YoloV3


# My libraries
from transf_utils import *

bridge = CvBridge() # bridge from opencv to ROS

class PlantDetectorNode:

    def __init__(self):
        
        super().__init__()
        rospy.init_node('plant_detector_node')
        
        # Publisher for the robot's target trajectory    
        self.target_pose_pub = rospy.Publisher('/smooth_trajectory', PoseArray, queue_size=10)

        # Publisher/Subscriber for "tf"
        self.tf_sub = tf.TransformListener()
        self.tf_pub = tf.TransformBroadcaster()

        # Subscriber to force feedback from "force_estimation_node.py"
        self.force_feedback_sub = rospy.Subscriber('force_estimation', String, self.force_feedback_callback)
        self.force_feedback = [] # init

        # Subscriber to the robot's "current_pose" (tf from base to ee)
        self.curr_pose_sub = rospy.Subscriber('/panda_force_ctrl/current_pose', PoseStamped, self.curr_pose_callback)
        self.curr_pose = [] # init current Panda robot Pose()

        # Subscribers to Yolo for general object detection (used for detecting the pottedplant)
        self.detection_image_sub = rospy.Subscriber('/darknet_ros/detection_image', Image, self.detection_image_callback)
        self.obj_img_cv2 = [] # init object detection image
        self.detection_bboxes_sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.detection_bboxes_callback)
        self.curr_frame_obj_bboxes = []  # init list of detected objects in the current frame, solidal to the camera

        # Subscriber to the Realsense2 "camera_info"
        self.camera_info_sub = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)
        
        # Subscriber to the Realsense2 "/camera/aligned_depth_to_color/image_raw"
        self.depth_img_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_img_callback)
        self.depth_img_cv2 = [] # init 

        # Subscriber to the Realsense2 "/camera/color/image_raw"
        self.rgb_img_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_img_callback)
        self.rgb_img = None # init 

        # Initialize keyboard Listener to detect keys inserted by user
        keyboard.Listener(on_press=self.press_callback).start()
        self.confirmation = False # init flag

        # Assign parameters  
        self.target_eul_1 = [3.010, 1.502, 2.040]
        self.target_eul_2 = [-2.890, 1.466, 1.344]

        self.save_images = True  # save all the images

        self.img_rgb_counter = 1        # counter for rgb images to be saved
        self.img_depth_counter = 1      # counter for depth images to be saved
        self.img_det_counter = 1        # counter for detection images to be saved
        self.img_strawdet_counter = 1   # counter for detection strawberry images to be saved
        self.detected_straw = False     # bool to know if strawberry has been detected

        self.image_w = 640  # raspberrypi's (dome) image width
        self.image_h = 480  # raspberrypi's (dome) image height
        self.intrinsic = []
        
        # Init state of the Finite State Machine
        # 1 = Find plant
        # 2 = 
        self.state = 1
        self.prev_state = 1

        # Hand-eye calibration output:
        self.T_ee_cam = np.eye(4)
        self.T_ee_cam[0,3] = -0.06251360103463763   # y on the hand-eye 
        self.T_ee_cam[1,3] = -0.015864171953963058 # x on the hand-eye
        self.T_ee_cam[2,3] = 0.038967573086997796  # z on the hand-eye
        self.T_ee_cam[0:3, 0:3] =   quat_2_rot(eul_2_quat(-np.pi/2,0,0))

        print("Hand-eye calibration output (T_ee_cam):\n", self.T_ee_cam)
        
        self.tf_plant_trasl = [] # init tf argument for plant frame
        self.tf_plant_quat = []  # init tf argument for plant frame
        self.target_1_base = []    # init plant boundaries to be used as target points
        self.target_2_base = []    # init plant boundaries to be used as target points
        self.traj_duration = 0           # duration [s] of the robot's trajectory 
        self.traj_finished = True  # bool to know if trajectory is finished
        self.traj_init_time = 0

        # Remove all previous logs 
        if os.path.exists(os.path.abspath(os.path.dirname(__file__)) + "/logs"):
            shutil.rmtree(os.path.abspath(os.path.dirname(__file__)) + "/logs", ignore_errors=True)
        os.mkdir(os.path.abspath(os.path.dirname(__file__)) + "/logs")

        for dir2make in ["rgb", "depth" "detection_yolo", "detection_plant", "detection_straw"]:
            os.mkdir(os.path.abspath(os.path.dirname(__file__)) + "/logs/" + dir2make)
        print("All logs folders created!")
       

        # Load Convolutional Neural Networks:
        # --------- STRAWBERRY ----------------
        model = 2 # choose the obj.det. model to find strawberries
        if model == 1:
            print("[1] Loading fine-tuned model from Yolo")
            self.straw_CNN = [] # init fine-tuned Yolo model for strawberry detection
            self.weights = os.path.abspath(os.path.dirname(__file__)) + "/checkpoints/model_20/"
            self.straw_CNN = YoloV3(classes=2)
            self.straw_CNN.load_weights(self.weights).expect_partial()

        elif model == 2:
            print("[2] Loading pre-trained Yolo tiny for strawberries")
            self.straw_CNN = cv2.dnn.readNet(os.path.abspath(os.path.dirname(__file__))+"/model/strawberry_yolotiny_2000.weights", os.path.abspath(os.path.dirname(__file__))+"/model/strawberry_yolotiny.cfg")

        self.ripe_straw_info = []
        self.straw_classes = ["Ripe_Straw", "Unripe_Straw"]
        
        # --------- PLANT ----------------
        # Warning: no need to load a CNN because it is launched using "roslaunch darknet_ros darknet_ros_rs.launch"
        self.straw_plant_info = []

        # Start control loop
        timer_period = 0.5 # [seconds]
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.control_loop) 
    # END init()

    # when user presses a key on the keyboard (also if the terminal is not currently selelcted!)
    def press_callback(self, key): 
        if key==keyboard.KeyCode.from_char('+'):
            print("[Keyboard] Confirmed action!")
            self.confirmation = True 
        
    def force_feedback_callback(self, msg): 
        self.force_feedback = np.array(msg)
        #print("TEST: receiving force:", self.force_feedback)
    
    def get_T_base_ee(self):
        # From current pose of the panda
        T_base_ee = np.eye(4)
        T_base_ee[:,3] = self.curr_pose[0], self.curr_pose[1], self.curr_pose[2], 1
        T_base_ee[:3, :3] = R.from_quat(self.curr_pose[3:]).as_matrix()
        #print("T_base_ee", T_base_ee) # to check

        return T_base_ee

    def get_T_base_cam(self):        
        # Computed transform from base to camera
        T_base_cam = self.get_T_base_ee().dot(self.T_ee_cam)   
        #print("T_base_cam", T_base_cam) # to check

        return T_base_cam

    def curr_pose_callback(self, msg): 
        self.curr_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                          msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]  
        #print("or", quat_2_eul(msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w))

    def camera_info_callback(self, camera_info):        
        self.intrinsic = rs.intrinsics()
        self.intrinsic.width = camera_info.width
        self.intrinsic.height = camera_info.height
        R = camera_info.K
        fx = R[0]
        fy = R[4]
        cx = R[2]
        cy = R[5]
        self.intrinsic.ppx = cx
        self.intrinsic.ppy = cy
        self.intrinsic.fx = fx
        self.intrinsic.fy = fy
        self.intrinsic.coeffs = camera_info.D
        
    # Function to zoom an image
    def zoom_img(self, img, zoom=1, angle=0, coord=None):
        if coord is None:
            cy, cx = [ i/2 for i in img.shape[:-1] ] 
        else:
            cy, cx = coord
            cy, cx = int(cy), int(cx)

        rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    
         
    def compute_trajectory(self, target_points, target_euls, traj_duration):  
        #-------------
        # Create instances: the Ruckig OTG as well as input and output parameters
        # Source: https://pypi.org/project/ruckig/
        dof = 3 # [x,y,z]
        otg = Ruckig(dof, 0.001)  # DoFs, control cycle
        inp = InputParameter(dof)
        out = OutputParameter(dof)

        # Init position and orientation
        last_position = self.curr_pose[:3]
        last_orientation = self.curr_pose[3:]


        # Create ROS message
        target_traj = PoseArray() # init
        target_traj.header.stamp = rospy.Time.now() 
        target_traj.header.frame_id = 'panda_link0' 

        for target_point,target_eul in zip(target_points, target_euls): 
            
            # Set input parameters
            inp.current_position = last_position
            inp.current_velocity = [0.0 for i in range(dof)]
            inp.current_acceleration = [0.0 for i in range(dof)]

            inp.target_position = [target_point[0], target_point[1], target_point[2]]
            inp.target_velocity = [0.0 for i in range(dof)]
            inp.target_acceleration = [0.0 for i in range(dof)]

            # Source: https://frankaemika.github.io/docs/control_parameters.html
            inp.max_velocity = [1.7, 2.5, 2.175]
            inp.max_acceleration = [13.0, 25.0, 10.0,] 
            inp.max_jerk = [12, 12, 12] 
                    
            # Set minimum duration (equals the trajectory duration when target velocity and acceleration are zero)
            inp.minimum_duration = traj_duration # [s]

            # Generate the trajectory within the control loop
            first_output, out_list = None, []
            res = Result.Working                          
            
            # Interpolate orientations:
            norm_curr_pose = last_orientation            
            
            target_quat = eul_2_quat(target_eul)
            quats = np.array([norm_curr_pose, target_quat])
            key_rots = R.from_quat(quats)
            key_times = [0, inp.minimum_duration] 

            slerp = Slerp(key_times, key_rots)

            times = np.linspace(0, inp.minimum_duration, num=10001)
            interp_rots = slerp(times)
          
            # Interpolate translations:
            ind = 0
            while res == Result.Working:
                res = otg.update(inp, out)
                out_list.append(copy(out))
                out.pass_to_input(inp)

                if not first_output:
                    first_output = copy(out)

                orientation = Quaternion()
                orientation.x, orientation.y, orientation.z, orientation.w = self.curr_pose[3:] #interp_rots.as_quat()[ind,:]
                ind += 1

                position = Point()           
                position.x, position.y, position.z = out.new_position[0], out.new_position[1], out.new_position[2]

                pose = Pose()
                pose.position = position
                pose.orientation = orientation 

                target_traj.poses.append(pose) # append to PoseArray ROS msg

                if ind == 10001:
                    last_orientation = interp_rots.as_quat()[-1,:]
                    last_position = [out.new_position[0], out.new_position[1], out.new_position[2]]
                    print("Assigned last position/orientation!")

            self.traj_duration = first_output.trajectory.duration                         
            print(f'Calculation duration: {first_output.calculation_duration:0.1f} [µs]')
            print(f'Trajectory duration: {self.traj_duration:0.4f} [s]')  


            # To plot traj:
            '''
            points_to_plot = np.array(points_to_plot)
            x,y,z = points_to_plot[:,0],points_to_plot[:,1],points_to_plot[:,2]
        
            # Plot trajectory
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            ax.scatter3D(x[:-1], y[:-1], z[:-1], color = "green") # all points
            ax.scatter3D(x[-1], y[-1], z[-1], color = "red") # target point

            #plt.title("Visualization of the robot's trajectory")
            #plt.show()
            '''
        return target_traj

        
    # control loop, executed each 0.2 seconds
    def control_loop(self, time):
      
        # Publish tf frames to visualize on RVIZ: 
        if self.tf_plant_trasl != []: # plantcenter_base
            self.tf_pub.sendTransform(self.tf_plant_trasl, self.tf_plant_quat, rospy.Time.now(), "plantcenter_base", "panda_link0")     
        if self.target_1_base != []: # target1cam_base
            self.tf_pub.sendTransform(self.target_1_base[:3], eul_2_quat(self.target_eul_1), rospy.Time.now(), "target1cam_base", "panda_link0")     
        if self.target_2_base != []: # target2cam_base
            self.tf_pub.sendTransform(self.target_2_base, eul_2_quat(self.target_eul_2), rospy.Time.now(), "target2cam_base", "panda_link0")     
        if self.curr_pose != []: # realsense_base
            trasl = self.get_T_base_cam()[:3, 3]
            quat = rot_2_quat(self.get_T_base_cam()[:3,:3])
            self.tf_pub.sendTransform(trasl, quat, rospy.Time.now(), "realsense_base", "panda_link0")


        # [State 1] Find plant
        if self.state == 1:
            if self.obj_img_cv2 == []:
                print("[State 1] Waiting for the object detection image...")                
            else: 
                print("[State 1] Find plant...")
                
                #cv2.imshow("Find plants", self.obj_img_cv2)
                #cv2.waitKey(1)
                
                if len(self.curr_frame_obj_bboxes) > 0: # if at least 1 object is found
                    for bb in self.curr_frame_obj_bboxes:
                        # CHECK IF THE OBJECT IS A PLANT --> "pottedplant", "vase"
                        if bb[0] == "pottedplant" or bb[0] == "vase" or bb[0] == "cup":                            
                            beep(sound='coin')
                            x1_px, x2_px, y1_px, y2_px = bb[3:7]                     
                            
                            # Compute min x1, x2, y1, y2 (3D) coordinates of the bounding box
                            #x1_y1_3d = rs.rs2_deproject_pixel_to_point(self.intrinsic, [y1_px, x1_px], self.depth_img_cv2[y1_px, x1_px]) 
                            #x1_y1_3d = [el/1000 for el in x1_y1_3d] # convert mm to m
                            #x2_y2_3d = rs.rs2_deproject_pixel_to_point(self.intrinsic, [y2_px, x2_px], self.depth_img_cv2[y2_px, x2_px]) 
                            #x2_y2_3d = [el/1000 for el in x2_y2_3d] # convert mm to m

                            plant_center_px = [int(x1_px+(x2_px-x1_px)/2-1), int(y1_px+(y2_px-y1_px-1)/2)]
                            plant_center_depth = self.depth_img_cv2[plant_center_px[1], plant_center_px[0]]
                        
                            plant_center_3d = rs.rs2_deproject_pixel_to_point(self.intrinsic, plant_center_px, plant_center_depth) 
                            plant_center_3d = [el/1000 for el in plant_center_3d] # convert mm to m
                            X_3d, Y_3d, Z_3d = plant_center_3d
                            distance = np.sqrt(X_3d**2 + Y_3d**2 + Z_3d**2)

                            print("[State 1] Detected plant! plant_center_3d:", plant_center_3d, "dist =", round(distance,4))

                            # ------------------------------          
                            # NEW PART TESTED AT HOME...
                            # ------------------------------          
                            
                            # Plant center from Σcamera to Σbase
                            plant_center_cam = [plant_center_3d[0], plant_center_3d[1], plant_center_3d[2], 1]
                            plant_center_base = self.get_T_base_cam().dot(np.array(plant_center_cam).T)
                            plant_center_base = plant_center_base[:3]        
                            print("Plant center wrt camera frame", plant_center_cam)
                            print("Plant center wrt base frame", plant_center_base)  

                            self.tf_plant_trasl = plant_center_base # compute tf translation
                            self.tf_plant_quat = self.curr_pose[3:] # compute tf orientation

                            # Compute the 2 target points that determine a diagonal trajectory (to test if it's enough)
                            th_z = plant_center_3d[2]/3  
                            th_x = 0.3
                            th_y = 0.06
                            target_1_cam = np.array([ plant_center_3d[0] + th_x,
                                                      plant_center_3d[1] + th_y,
                                                      plant_center_3d[2] - th_z,
                                                      1])
                            target_2_cam = np.array([ plant_center_3d[0] - th_x,
                                                      plant_center_3d[1] + th_y,
                                                      plant_center_3d[2] - th_z,
                                                        1])

                          

                            print("[State 1] target_1_cam", target_1_cam)
                            print("[State 1] target_2_cam", target_2_cam)

                            # Target 1 from Σcamera to Σbase, so that the camera goes to target!
                            print("self.T_ee_cam.dot(target_1_cam.T)", self.T_ee_cam.dot(target_1_cam.T))
                            print("-")
                            print("self.T_ee_cam[:,3]", self.T_ee_cam[:,3])
                            #target_1_ee = self.T_ee_cam.dot(target_1_cam.T) - self.T_ee_cam[:,3]
                            #print("[State 1] target_1_ee", target_1_ee)
                            
                            
                            self.target_1_base = self.get_T_base_cam().dot(target_1_cam) #self.get_T_base_ee().dot(target_1_ee)
                            self.target_1_base = self.target_1_base[:3]
                            print("[State 1] target_1_base:", self.target_1_base)

                            # Target 2 from Σcamera to Σbase, so that the camera goes to target!
                            #target_2_ee = self.T_ee_cam.dot(target_2_cam.T) - self.T_ee_cam[:,3]
                            self.target_2_base = self.get_T_base_cam().dot(target_2_cam)
                            self.target_2_base = self.target_2_base[:3]
                            print("[State 1] target_2_base:", self.target_2_base)

                            self.state = 2 # switch to next state
                            self.prev_state = 1
                           
        
                        
        # [State 2] Scan plant starting from first target point (plant's left boundary)
        elif self.state == 2:
            print("[State 2] Scan plant starting from first target point (plant's left boundary)...")
            print("[State 2] Plant info:", self.straw_plant_info)

            plant_center_3d = self.straw_plant_info
          
            if self.confirmation == False:                
                print("[State 2] Press '+' to confirm plant approach!")
                print("[State 2] Current robot pose:", [round(el,4) for el in self.curr_pose])
                print("[State 2] Target 1 (wrt base):", self.target_1_base)
                print("[State 2] Target 2 (wrt base):", self.target_2_base)             
            else: # after confirmation                        
                # Generate and publish the trajectory to go to the first target                
                # target_eul = [-1.9928173567952219, 1.509207099637598, -1.1142493702219025] 
                # target_traj = self.compute_trajectory(self.curr_pose[:3], target_quat)

                #print("Target_quat in eul:", [el*180/np.pi for el in quat_2_eul(target_quat)])

                target_euls = [self.target_eul_1, self.target_eul_2]

                target_points = [self.target_1_base, self.target_2_base]

                target_traj = self.compute_trajectory(target_points, target_euls, 10)

                self.target_pose_pub.publish(target_traj)
                self.traj_init_time = rospy.get_time() # in seconds

                print("[State 2] Trajectory to target_1_base has been published!")
                self.state = 3 # switch to straw detection state
                self.prev_state = 2
                self.confirmation = False # init. for safety reasons                


        # [State 3] Looking for strawberries while robot is moving
        elif self.state == 3:
            print("[State 3] Looking for strawberries while robot is moving...")

            # Detect strawberries on RGB images       
            layer_names = self.straw_CNN.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in self.straw_CNN.getUnconnectedOutLayers()]
            colors = [(0,0,255), (0,255,0)]

            if self.rgb_img is None:
                print("[State 3] Not receiving RGB images!")
            else:
                zoom_factor = 2
                img = self.zoom_img(self.rgb_img.copy(), zoom=zoom_factor)
                height, width, _ = img.shape

                # Detecting objects
                blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
                self.straw_CNN.setInput(blob)
                
                # Get CNN outputs
                outs = self.straw_CNN.forward(output_layers)

                # Showing informations on the screen
                raw_img = self.rgb_img.copy()
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        label = str(self.straw_classes[class_id]) 

                        if confidence > 0.05 and label == "Ripe_Straw":
                            beep(sound='coin') #success detecting straw

                            # Coordinates on the zoomed frame
                            center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                            w, h = int(detection[2] * width), int(detection[3] * height)
                            x, y = int(center_x - w / 2), int(center_y - h / 2)       

                            # Draw on zoomed frame
                            cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_id], 2)
                            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, colors[class_id], 3)
                            cv2.circle(img, (center_x, center_y), 5, colors[class_id], -1)
            


                            
                            # Recompute coordinates on the non-zoomed frame
                            w, h = int(w/zoom_factor), int(h/zoom_factor)
                            center_x = int(width/zoom_factor/2 + detection[0]*width/zoom_factor)
                            center_y = int(height/zoom_factor/2 + detection[1]*height/zoom_factor)
                            x, y = int(center_x - w / 2), int(center_y - h / 2)
                        
                            # Compute 3D coordinates (center and boundaries)
                            x1_y1_3d = rs.rs2_deproject_pixel_to_point(self.intrinsic, [y, x], self.depth_img_cv2[y, x]) 
                            x1_y1_3d = [el/1000 for el in x1_y1_3d] # convert mm to m
                            x2_y2_3d = rs.rs2_deproject_pixel_to_point(self.intrinsic, [y+h, x+w], self.depth_img_cv2[y+h, x+w]) 
                            x2_y2_3d = [el/1000 for el in x2_y2_3d] # convert mm to m

                            straw_center_depth = self.depth_img_cv2[center_y, center_x]
                            straw_center_3d = rs.rs2_deproject_pixel_to_point(self.intrinsic, [center_x, center_y], straw_center_depth) 
                            straw_center_3d = [el/1000 for el in straw_center_3d] # convert mm to m

                            self.ripe_straw_info = [straw_center_3d, x1_y1_3d, x2_y2_3d]
                            print("[Strawberry detection] straw info wrt camera (straw_center_3d, x1_y1_3d, x2_y2_3d):", self.ripe_straw_info)

                            # Draw on non-zoomed frame
                            cv2.rectangle(raw_img, (x, y), (x + w, y + h), colors[class_id], 2)
                            cv2.putText(raw_img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, colors[class_id], 3)
                            cv2.circle(raw_img, (center_x, center_y), 5, colors[class_id], -1)

                            self.confirmation = False # re-init. for safety reasons
                            self.detected_straw = True


                # Show detection
                #cv2.imshow("Strawberry cropped", img)
                #cv2.imshow("Strawberry full", raw_img)
                #cv2.waitKey(1)            

                if self.save_images:
                    path = os.path.abspath(os.path.dirname(__file__)) + "/logs/detection_straw/" + str(self.img_strawdet_counter)+"_zoomed.png"
                    cv2.imwrite(path, img)
                    path = os.path.abspath(os.path.dirname(__file__)) + "/logs/detection_straw/" + str(self.img_strawdet_counter)+"_raw.png"
                    cv2.imwrite(path, raw_img)

            # FSM logic:
            # straw detected [3] --> [5] approach straw
            if self.detected_straw: 
                print("[State 3] Strawberry was found! Switching to state 5 (approach)")
                self.state = 4  # change to strawberry approach state
                self.prev_state = 3

                target_traj = self.compute_trajectory([self.curr_pose[:3]], [quat_2_eul(self.curr_pose[3:])], 1)
                self.target_pose_pub.publish(target_traj)


        # [State 4] Scan plant starting from second target point (plant's right boundary)
        elif self.state == 4:
            print("[State 4] Approach the ripe strawberry")

            if self.confirmation == False:                
                print("[State 4] Press '+' to confirm strawberry approach!")                                      
            else: # after confirmation  
                # Generate and publish the trajectory to go to the first target
                straw_center_3d, x1_y1_3d, x2_y2_3d = self.ripe_straw_info
                
                # Compute the 2 target points that determine a diagonal trajectory (to test if it's enough)
                th_x = 0
                th_y = 0.06
                th_z = 0.16
                target_straw_cam = np.array([ straw_center_3d[0] - th_x,
                                              straw_center_3d[1] - th_y,
                                              straw_center_3d[2] - th_z,
                                              1])

                self.target_straw_base = self.get_T_base_cam().dot(np.array(target_straw_cam).T)
                self.target_straw_base = self.target_straw_base[:3]      
                
                print("[State 4] target_straw_base:", self.target_straw_base)

                target_eul = [1.849, 1.523, 0.332]    
                target_traj = self.compute_trajectory([self.target_straw_base], [target_eul], 10)
                self.target_pose_pub.publish(target_traj)

                print("[State 4] Trajectory to target_straw_base has been published!")
                self.state = 5 # switch to straw detection state
                self.prev_state = 4
                self.confirmation = False # init. for safety reasons                
     
      

        elif self.state == 5:
            print("End....")
            pass
            '''
            print("[State 6] Close the gripper and produce force feedback")
            print("[State 6] [Warning] force_estimation_node.py has to run!")
            print("[State 6] Listening to force feedback:", self.force_feedback)

            _, _, Fz_hat = self.force_feedback
            if abs(Fz_hat) < 1.7: # Force threshold to stop the closing motion [N]
                # TODO: INSERT CODE to CLOSE THE GRIPPER
                # rostopic pub /frank_gripper/move/goal frank_gripper/MoveActionGoal "header... 
                print("[State 6] Keep closing!")
                # https://projects.saifsidhik.page/panda_robot/
                # https://github.com/justagist/panda_robot
            else:
                # TODO: INSERT CODE to STOP GRIPPER
                # rostopic pub /frank_gripper/move/goal frank_gripper/MoveActionGoal "header... 
                print("[State 6] STOP CLOSURE!!")
                self.state = 6 # go to next state: picking pattern to harvest straw
                self.prev_state = 5 # go to next state: picking pattern to harvest straw

                self.confirmation = False # re-init. for safety reasons
            '''

        elif self.state == 7:
            print("[State 7] Harvest the ripe strawberry...")

            if self.confirmation == False:
                print("[Warning] Press '+' to confirm closure!")
            else:

                # TODO: INSERT CODE to MOVE THE ROBOT IN A CERTAIN TRAJ TO HARVEST STRAW
                # del tipo: - go down 5 cm while rotating,  go back 5 cm ???
                pass


        elif self.state == 10: # JUST A TEST STATE TO MOVE THE ROBOT
            print("[State 10] TEST to move...")

            if self.confirmation == False:
                print("[Warning] Press '+' to confirm!")
            else:
                # Generate and publish the trajectory to go to the first target  
                target_quat = [self.curr_pose[3]+0.2, self.curr_pose[4], self.curr_pose[5], self.curr_pose[6]]  #norm_quat(-0.3411435180913495, 0.6457974253132484, 0.27094340231705016, 0.6270173412330675)
                target_pos = [self.curr_pose[0]+0.1, self.curr_pose[1], self.curr_pose[2]]   #self.curr_pose[:3]
                target_traj = self.compute_trajec02ime() # in seconds

                print("[State 10] Trajectory to target_1_base has been published!")
                self.state = 11 # switch to straw detection state
                self.prev_state = 10
                self.confirmation = False # init. for safety reasons 
                exit()               

        elif self.state == 11:
            print("State 11.....")


    def detection_image_callback(self, img):                
        self.obj_img_cv2 = bridge.imgmsg_to_cv2(img)  #convert ROS to OpenCV

        if self.save_images:
            # Save general obj. det. image
            path = os.path.abspath(os.path.dirname(__file__)) + "/logs/detection_yolo/" +str(self.img_det_counter)+".png"
            cv2.imwrite(path, self.obj_img_cv2)  

            # Save image with only the plant bbox
            path = os.path.abspath(os.path.dirname(__file__)) + "/logs/detection_plant/" +str(self.img_det_counter)+".png"
            img_plant = self.rgb_img.copy()

            if len(self.curr_frame_obj_bboxes) > 0:
                for bb in self.curr_frame_obj_bboxes:
                    # CHECK IF THE OBJECT IS A PLANT --> "pottedplant", "vase"
                    if bb[0] == "pottedplant" or bb[0] == "vase":                            
                        x1_px, x2_px, y1_px, y2_px = bb[3:7]  

                        # Draw on zoomed frame
                        cv2.rectangle(img_plant, (x1_px, y1_px), (x2_px, y2_px), [0,255,0], 2)
                        cv2.putText(img_plant, "Plant", (x1_px, y1_px+30), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,0], 2)
                        cv2.imwrite(path, img_plant) 
            self.img_det_counter += 1


    def rgb_img_callback(self, img):        
        bgr_img = bridge.imgmsg_to_cv2(img)  #convert ROS to OpenCV
        self.rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)    

        if self.save_images:
            path = os.path.abspath(os.path.dirname(__file__)) + "/logs/rgb/" +str(self.img_rgb_counter)+".png"
            cv2.imwrite(path, self.rgb_img)    
            self.img_rgb_counter += 1
        
    def depth_img_callback(self, img):   
        bgr_img = bridge.imgmsg_to_cv2(img)  #convert ROS to OpenCV
        self.depth_img_cv2 = bgr_img

    def detection_bboxes_callback(self, msg):
        msg_bboxes = msg.bounding_boxes
        self.curr_frame_obj_bboxes = [] # init list of lists

        for msg_bbox in msg_bboxes:            
            # to avoid out of bounds exception:
            if msg_bbox.xmin > 0: 
                msg_bbox.xmin -= 1
            if msg_bbox.xmax > 0:
                msg_bbox.xmax -= 1
            if msg_bbox.ymax > 0:
                msg_bbox.ymax -= 1
            if msg_bbox.ymin > 0:
                msg_bbox.ymin -= 1
            # ---               
            self.curr_frame_obj_bboxes.append([msg_bbox.Class, msg_bbox.id, msg_bbox.probability, msg_bbox.xmin, msg_bbox.xmax, msg_bbox.ymin, msg_bbox.ymax])



 

# START main()
if __name__ == '__main__':
   
    plant_detector_node = PlantDetectorNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("ROSInterruptException!")

    rospy.shutdown()
    





