#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import os
import joblib

import time
import numpy as np
import open3d as o3d 
import pyrealsense2 as rs
from sklearn import linear_model # Robustly fit linear model with RANSAC algorithm


# Import Python libraries
import cv2
from sklearn.metrics import mean_squared_error

import os
import pandas as pd
import numpy as np
import socket
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import pickle
from sklearn import preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from pynput import keyboard

# Import my libraries
from connection_to_client import read_camera_frame, connect_to_server
from camera_params import *
from online_tracking import *
from fit_3d_surface import create_mesh_from_points, fit_3d_surface_main
from colors import *

from visualize_results import *

import time
from keras.models import model_from_json
from scipy import signal



class ForceEstimator:

    def __init__(self):
        super().__init__()

        rospy.init_node('force_estimation')

        self.force_feedback_pub = rospy.Publisher('force_estimation', String, queue_size=10)

        # Fitting algorithm:
        self.fitted_reference_lines = False
        self.central_point = [] 
        self.vert_points  = []
        self.hor_points = []
        self.diag_neg_points = [] 
        self.diag_pos_points = []

        # Algorithm params:
        self.img_width = -1
        self.img_height = -1
        self.num_markers_gripper = 29 
        self.marker_shape = 'circle'
        self.distance_th = 10  # min. threshold distance before interpolating trajectories (u-v-radius)
        self.filter_cutoff_freq = 5  # lpf params
        self.filter_order = 2  # lpf params
        Ts = 0.024  # (all_times[-1]-all_times[-2])/1000000000/2
        #print(Ts)  # to tune the value and avoid recomputing it every time / changing the filtering output
        self.fs = 1 / Ts  # frequency

        self.app_point_data = []
        self.Ftot_preds_act = []

        # Get the features for ML methods in the same way:
        self.feature_option = 3
        self.feature_scaling = False
        self.models_folder = os.path.abspath(os.path.dirname(__file__)) + "/fitted_models/feat"+str(self.feature_option)+"_NOscaling/"
        print("Looking for pretrained models in dir:", self.models_folder)

        # Force estimation params:
        self.model_names = ['My_old', 'My_new', 'SVR', 'KNN', 'LR', 'CNN']

        # 1a - [Cx_hat_final, Cy_hat_final, Cz_hat_final] from "my_force_estimation_pixel.py"
        self.my_method_coeffs = [-0.03380031936618423, 0.018187694851045568, -0.772042465826837]
        print("[1a estimation method], linear coefficients:", self.my_method_coeffs)

        # 1b - [Cx_hat_final, Cy_hat_final, Cz_hat_final] considering the mean of the 29 markers' coeffs. 
        # (in theory I would need to apply RANSAC etc to get every marker, but online could be unreliable and there's not time....)
        df_mymethod = pd.read_json(self.models_folder+'my_estimation_method.json')
        self.Cx_hat = np.mean(df_mymethod['Cx_hat'])
        self.Cy_hat = np.mean(df_mymethod['Cy_hat'])
        self.Cz_hat = np.mean(df_mymethod['Cz_hat'])
        print("[1b estimation method], mean of the 29 linear coefficients:", [self.Cx_hat, self.Cy_hat, self.Cz_hat])

        # 2 - Load SVR models
        print(self.models_folder + 'fitted_SVR_x')
        self.SVR_x = pickle.load(open(self.models_folder + 'fitted_SVR_x.pkl', 'rb'))
        self.SVR_y = pickle.load(open(self.models_folder + 'fitted_SVR_y.pkl', 'rb'))
        self.SVR_z = pickle.load(open(self.models_folder + 'fitted_SVR_z.pkl', 'rb'))

        # 3 - Load CNN model (from json and weigths)
        json_file = open(self.models_folder+'fitted_CNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.CNN = model_from_json(loaded_model_json)
        # load weights into new model
        self.CNN.load_weights(self.models_folder+"fitted_CNN.h5")
        self.CNN.compile(optimizer='adam', loss='mse', metrics=['mse'])

        # 4 - Load KNN model
        self.KNN = pickle.load(open(self.models_folder + 'fitted_KNN.pkl', 'rb'))

        # 5 - Load LR model
        self.LR = pickle.load(open(self.models_folder + 'fitted_LR.pkl', 'rb'))

        # Connect to raspberry
        self.raspberry_ip = "192.168.137.204" #"169.254.35.229" # #socket.gethostbyname("raspberrypifabio")  #"192.168.137.204" #
        port = 5550
        print("Raspberrypi IP address:", self.raspberry_ip)
        self.client_socket, self.payload_size, self.data = connect_to_server(self.raspberry_ip, port)

        # Start algorithm:
        print("Starting online force estimation pipeline...")

        self.initialized = False  # required to correctly store the initial markers in the actual correct number
        self.all_traj_markers = []  # initialize all detected markers' list
        self.prev_coords = []  #init previous markers coords.

        self.Ftot_preds_all = []
        self.json_index = 1 #index of json files
        self.show_prints = False # to print estimates etc..

        # Remove all logs
        logs_dir = "/home/fab/catkin_ws/src/tesi_gripper/src/logs/"
        for folder in [logs_dir+"force", logs_dir+"traj", logs_dir+"app_point", logs_dir+"img_frame_with_app_point",
                       logs_dir+"img_frame_with_markers", logs_dir+"img_raw_frame"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)


        self.raw_frame = []
        self.frame_with_markers = []
        self.frame_with_app_point = []

        # Initialize keyboard Listener to detect keys inserted by user
        listener = keyboard.Listener(on_press=self.press_callback)
        listener.start()

        self.timer = rospy.Timer(rospy.Duration(Ts), self.control_loop)


    def write_jsons(self):
        if len(self.Ftot_preds_all) > 0 and len(self.all_traj_markers) > 0:
            # Compute displacements by removing the initial bias to the markers' trajectories
            act_traj  = np.array(self.all_traj_markers)[-1,:,:]
            init_traj = np.array(self.all_traj_markers)[0,:,:]
            act_time  = np.array([time.time_ns(),time.time_ns(),time.time_ns()])


            #print("[2 - Force estimation] Shape pixel displacements:", px_disps.shape)
            px_disps = act_traj - init_traj

            #show_markers_displacements(self.all_times, px_disps)

            # Write forces inside json
            json_dir = "/home/fab/catkin_ws/src/tesi_gripper/src/logs/force/"
            json_data = np.array(self.Ftot_preds_act)
            df_json = pd.DataFrame(json_data, index=self.model_names, columns=["Fx_hat", "Fy_hat", "Fz_hat"])
            df_json.to_json(json_dir +str(self.json_index)+"_estimated_forces.json", orient="index")

            # Write trajectories inside json
            json_dir = "/home/fab/catkin_ws/src/tesi_gripper/src/logs/traj/"
            df_json = pd.DataFrame([act_time, act_traj.T, px_disps.T], index=["actual_timestamp", "actual_markers_loc",  "actual_markers_disp"])
            df_json.to_json(json_dir +str(self.json_index)+"_trajectories.json", orient="index")

            # Write application point inside json
            json_dir = "/home/fab/catkin_ws/src/tesi_gripper/src/logs/app_point/"
            df_json = pd.DataFrame(self.app_point_data, index=["cx", "cy",  "radius"])
            df_json.to_json(json_dir +str(self.json_index)+"_app_point.json", orient="index")

            # Save raw_frame 
            img_dir = "/home/fab/catkin_ws/src/tesi_gripper/src/logs/img_raw_frame/"
            cv2.imwrite(img_dir + str(self.json_index) +"_raw_frame.jpg", self.raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])   

            # Save frame_with_markers
            img_dir = "/home/fab/catkin_ws/src/tesi_gripper/src/logs/img_frame_with_markers/"
            cv2.imwrite(img_dir + str(self.json_index) +"_frame_with_markers.jpg", self.frame_with_markers, [cv2.IMWRITE_JPEG_QUALITY, 100])   

            # Save frame_with_app_point
            img_dir = "/home/fab/catkin_ws/src/tesi_gripper/src/logs/img_frame_with_app_point/"
            cv2.imwrite(img_dir + str(self.json_index) +"_frame_with_app_point.jpg", self.frame_with_app_point, [cv2.IMWRITE_JPEG_QUALITY, 100])   

            self.json_index += 1
            #print("[2 - Force estimation] Estimating forces...")
        else:
            print("[Warning] Not writing force logs because not initialized!")        


    
    def control_loop(self, t):
        raw_frame, frame, self.data = read_camera_frame(self.client_socket, self.payload_size, self.data, undistort_frame=False)
        cv2.imshow("Receiving video frame", frame)  # show raw frame from raspberry            
        cv2.waitKey(1)

   

        if self.img_width == -1:
            self.img_width = raw_frame.shape[1]
        if self.img_height == -1:
            self.img_height = raw_frame.shape[0]

        if self.show_prints:
            print("[1 - Online tracking] Tracking started!")            

       
        frame_with_markers, markers, num_detected_markers = detect_markers(raw_frame.copy(), self.marker_shape)
        cv2.imshow("Frame with markers", frame_with_markers)  # show raw frame from raspberry
        cv2.waitKey(1)


        self.raw_frame = raw_frame.copy()
        self.frame_with_markers = frame_with_markers.copy()

        if not self.initialized:  # check if initialization of initial markers was done
            if num_detected_markers == self.num_markers_gripper:
                self.all_traj_markers = [markers] # this way I can reset the bias!
                self.prev_coords = markers
                if self.show_prints:
                    print("[1 - Online tracking] Detected", num_detected_markers, "markers!")
                    print("[1 - Online tracking] Initialization done!")
                self.initialized = True
            else:
                print("[Warning] Trying to initialize marker detection...", num_detected_markers, "detected",  self.num_markers_gripper, "are required!")

        elif self.initialized:
            traj_update = []
            for marker_id in range(self.num_markers_gripper):
                # compute distances between old center and all new markers to see what's the closest
                old_center = np.array((self.prev_coords[marker_id][0], self.prev_coords[marker_id][1]))

                eucl_dists = []
                for m in markers:
                    new_center = np.array((m[0], m[1]))
                    eucl_dists.append(np.linalg.norm(old_center - new_center))

                if min(eucl_dists) < self.distance_th:  # impose threshold (else means marker was not detected in actual frame)
                    argmin_id = np.argmin(np.array(eucl_dists))

                    traj_update.append(markers[argmin_id])
                    # print(new_markers[argmin_id], "is close to", self.prev_coords[marker_id])
                else:
                    traj_update.append(self.prev_coords[marker_id])  # append the old values (already filtered)
                    # print("Marker is FAR from others, adding", self.prev_coords[marker_id])

            self.all_traj_markers.append(traj_update)
            self.prev_coords = traj_update

            all_traj_markers_arr = np.array(self.all_traj_markers)

            # Low pass filtering of pixel coordinates
            if self.show_prints:
                print("[2 - Online tracking] Filtering coordinates...")
            if all_traj_markers_arr.shape[0] > 10:
                for m_id in range(self.num_markers_gripper):
                    for coord_id in range(3):
                        all_traj_markers_arr[:, m_id, coord_id] = butter_lowpass_filter(
                            all_traj_markers_arr[:, m_id, coord_id], cutoff=self.filter_cutoff_freq, fs=self.fs,
                            order=self.filter_order)
            self.all_traj_markers = list(all_traj_markers_arr)

            if self.show_prints:
                print("[2 - Online tracking] Detected", num_detected_markers, "markers!")
            #print("all_traj_markers", np.array(all_traj_markers).shape)

                
            # Estimate application point and overlay it on the frame
            # To estimate the application point compute the average coords. of the bigger markers
            all_traj_markers_arr = np.array(self.all_traj_markers)
            actual_disps = all_traj_markers_arr[-1, :, :] - all_traj_markers_arr[0, :, :]

            #mean_radius_displacement = np.mean(abs(actual_disps[:, 2]))
            #print("mean_radius_displacement", mean_radius_displacement, mean_radius_displacement.shape)

            corner_points = []  # u,v coords. of markers that moved
            for m_id, radius_disp in enumerate(actual_disps[:, 2]):
                #if radius_disp > mean_radius_displacement*1.2:
                if radius_disp > 0.5: # 1 pixel of radius displacement
                    # print(radius_disp, ">", mean_radius_displacement*0.9)
                    corner_points.append([all_traj_markers_arr[0, m_id, 0],
                                            all_traj_markers_arr[0, m_id, 1]])
            if len(corner_points) > 0:
                corner_points = np.array(corner_points)
                estimated_app_point = [np.mean(corner_points[:, 0]),
                                        np.mean(corner_points[:, 1])]
            else:
                estimated_app_point = np.array([frame.shape[1]/2, frame.shape[0]/2])

            u, v = int(estimated_app_point[0]), int(estimated_app_point[1])
            rad = 2
            frame_with_app_point = frame.copy()
            # Draw estimated application point
            cv2.circle(frame_with_app_point, (u, v), rad, (0, 255, 0), 5)

            # Draw estimated application area (corners/fitted ellipse/fitted circle)
            if len(corner_points) >= 3:
                pts = np.array(corner_points, np.int32)
                (cx, cy), radius = cv2.minEnclosingCircle(pts)
                cv2.circle(frame_with_app_point, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)

                self.app_point_data = [cx, cy, radius]
            cv2.imshow("Frame with estimated application point", frame_with_app_point)
            cv2.waitKey(1)

            self.frame_with_app_point = frame_with_app_point.copy()


            # --------- end app. point estimation ---------- #

            # Having partial metric displacements for each marker, compute total force components:
            Ftot = np.array([0, 0, 0])
            Cx, Cy, Cz = self.my_method_coeffs # get prev. estimated coefficients

            all_traj_markers_arr = np.array(self.all_traj_markers) # compute displacements from start
            actual_disps = all_traj_markers_arr[-1, :, :] - all_traj_markers_arr[0, :, :]

            for disp in actual_disps:
                dx, dy, _ = disp
                F = np.array([Cx*dx, Cy*dy, Cz*np.sqrt(dx**2 + dy**2)])
                Ftot = np.sum([Ftot, F], axis=0)
            Ftot_mymethod = Ftot/self.num_markers_gripper

            if self.show_prints:
                print("[3a - Force estimation my_method] Fx,Fy,Fz =", Ftot_mymethod)

            Ftot_new = np.array([0, 0, 0])
            for disp in actual_disps:
                dx, dy, _ = disp
                F = np.array([self.Cx_hat*dx, self.Cy_hat*dy, self.Cz_hat*np.sqrt(dx**2 + dy**2)])
                Ftot_new = np.sum([Ftot_new, F], axis=0)
            Ftot_mymethod_otherfeat = Ftot_new/self.num_markers_gripper

            if self.show_prints:
                print("[3a2 - Force estimation my_method_otherfeat] Fx,Fy,Fz =", Ftot_mymethod_otherfeat)

            # Get the features for ML methods in the same way:
            

            # Option 1: for each ground truth, mean of all markers displacements U and V coords.
            if self.feature_option == 1:
                feat = [np.mean(actual_disps[:, 0]),
                                    np.mean(actual_disps[:, 1])]

            # Option 2: for each ground truth, mean of all markers displacements U, V coords. and pixel radius
            if self.feature_option == 2:
                feat = [np.mean(actual_disps[:, 0]),
                                    np.mean(actual_disps[:, 1]),
                                    np.mean(abs(actual_disps[:, 2]))]

            if self.feature_option == 3:
                feat = np.array([np.mean(abs(actual_disps[:, 0])),
                                    np.mean(abs(actual_disps[:, 1]))])

            if self.feature_option == 5:                    
                show_plots = True

                if not self.fitted_reference_lines:
                    ransac_th = 0.1 # required for sorting trajectories
                    coords_th = 25  # required for sorting trajectories

                    markers_x = all_traj_markers_arr[0, :, :][:,0]
                    markers_y = all_traj_markers_arr[0, :, :][:,1]
                    _, _, _, _, self.central_point, self.vert_points, self.hor_points, \
                    self.diag_neg_points, self.diag_pos_points = self.fit_reference_lines(ransac_th, coords_th, markers_x, markers_y, show_plots)
                    self.fitted_reference_lines = True

                    sorted_markers_traj = self.sort_markers_trajectory(all_traj_markers_arr, self.central_point, self.vert_points,self.hor_points, self.diag_neg_points, self.diag_pos_points, show_plots)
                    self.all_traj_markers = [sorted_markers_traj.tolist()]
                    print("self.prev_coords", self.prev_coords, sorted_markers_traj)
                    self.prev_coords = sorted_markers_traj.tolist()


                    feat = sorted_markers_traj[:, 0:2].flatten()
                
                else:
                    #feat = sorted_markers_traj[:, 0:2].flatten()
                    feat = np.array(self.all_traj_markers)[-1,:,0:2].flatten()

            # scaler = preprocessing.StandardScaler().fit(all_train_features)
            feat = np.array(feat)
            feat = np.reshape(feat, (-1, feat.shape[0]))
            
            if self.feature_scaling:
                scaler = joblib.load(self.models_folder + "fitted_MinMaxScaler.save")
                feat = scaler.transform(feat)

            # Compute predictions
            picking_task = True
            if picking_task:
                Ftot_SVR = np.array([0,0,0])
                Ftot_KNN = self.KNN.predict(feat)[0]
                Ftot_LR = np.array([0,0,0])
                Ftot_CNN = np.array([0,0,0])
            else:
                Ftot_SVR = np.array([self.SVR_x.predict(feat)[0], self.SVR_y.predict(feat)[0], self.SVR_z.predict(feat)[0]])
                Ftot_KNN = self.KNN.predict(feat)[0]
                Ftot_LR = self.LR.predict(feat)[0]
                Ftot_CNN = self.CNN.predict(feat, verbose=0)[0] #[0,0,0] #

            if self.show_prints:
                print("[3b - Force estimation SVR] Fx,Fy,Fz =", Ftot_SVR)
                print("[3c - Force estimation KNN] Fx,Fy,Fz =", Ftot_KNN)
                print("[3d - Force estimation LR] Fx,Fy,Fz =", Ftot_LR)
                print("[3e - Force estimation CNN] Fx,Fy,Fz =", Ftot_CNN)

            frame_with_force = frame.copy()
            y_pos = 20
            x_pos = 20
            self.Ftot_preds_act = [Ftot_mymethod, Ftot_mymethod_otherfeat, Ftot_SVR, Ftot_KNN, Ftot_LR, Ftot_CNN]
            self.Ftot_preds_all.append([el.tolist() for el in self.Ftot_preds_act]) #collect all predictions

            for Ftot,model_name in zip(self.Ftot_preds_act, self.model_names):
                text = model_name + " Fx:"+str(round(Ftot[0],2)) + "; Fy:"+str(round(Ftot[1],2)) + "; Fz:"+str(round(Ftot[2],2))

                if picking_task:
                    if model_name == "KNN":
                        x_pos = 40
                        y_pos = 220
                        frame_with_force = cv2.putText(frame_with_force, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                else:
                    y_pos += 30
                    frame_with_force = cv2.putText(frame_with_force, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow("Receiving video frame + forces", frame_with_force)  # show raw frame from raspberry
            cv2.waitKey(1)
        
            self.write_jsons() #inside the "if self.initialized"

            # Publish string containing force feedback generated by KNN
            self.force_feedback_pub.publish(str(Ftot_KNN))

        # end  "if self.initialized"
 
    # end control_loop()

            
        

    # when user presses a key on the keyboard (also if the terminal is not currently selelcted!)
    def press_callback(self, key): 
        if key==keyboard.KeyCode.from_char('r'): #careful: this messes up with logging forces and trajs!!!
            print("Resetting bias...")
            self.initialized = False #this way the estimation will restart from zero, removing any bias
        
    # Function to get the index of the closest point inside a np.array of points (ex shape: 9x2)
    def closest_point(self, points_arr, single_point_arr):
        dist_2 = np.sum((points_arr - single_point_arr) ** 2, axis=1)
        return np.argmin(dist_2)



    def fit_reference_lines(self, ransac_th, coords_th, markers_x, markers_y, show_plots):
        dyn_increment = 0.1

        vertical_line, horizontal_line, diag_neg_line, diag_pos_line = [], [], [], []
        central_point = []
        vertical_points, horizontal_points, diag_neg_points, diag_pos_points = [], [], [], []


        # 1 - Find vertical line
        num_inliers = 0
        ransac_th_dynamic = ransac_th

        x_to_fit, y_to_fit = [], []
        for x, y in zip(markers_x, markers_y):
            if abs(x - self.img_width / 2) < coords_th:
                x_to_fit.append(x)
                y_to_fit.append(y)
        x_to_fit = np.array(x_to_fit).reshape(-1, 1)
        y_to_fit = np.array(y_to_fit)



        while num_inliers != 9:
            # Define RANSAC model
            ransac = linear_model.RANSACRegressor(max_trials=10000, stop_n_inliers=9, residual_threshold=ransac_th_dynamic)

            ransac.fit(x_to_fit, y_to_fit)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            num_inliers = x_to_fit[inlier_mask].shape[0]
            print("Number of inliers", num_inliers)

            if num_inliers == 9:
                print("Found vertical line!")
                vertical_line = [ransac.estimator_.coef_[0], ransac.estimator_.intercept_]
                vertical_points = np.array([x_to_fit[:, 0][inlier_mask], y_to_fit[inlier_mask]]).T

                # Find the central point and remove it from "vertical_points" array
                #center_id = self.closest_point(vertical_points, np.array([self.img_width/2, self.img_height/2]))
                #central_point = vertical_points[center_id,:]
                if len(central_point) == 0:
                    columnIndex = 1
                    vertical_points = vertical_points[vertical_points[:, columnIndex].argsort()]
                    center_id = 4
                    central_point = vertical_points[center_id, :]
                vertical_points = np.delete(vertical_points, center_id, 0)
            else:
                ransac_th_dynamic += dyn_increment

        if show_plots:
            plt.scatter(x_to_fit[:,0][inlier_mask], y_to_fit[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
            plt.scatter(x_to_fit[:,0][outlier_mask], y_to_fit[outlier_mask], color="gold", marker=".", label="Outliers")
            plt.legend(loc="lower right")
            plt.xlabel("u (pixels)")
            plt.ylabel("v (pixels)")
            plt.show()

        # 2 - Find horizontal line
        ransac_th_dynamic = ransac_th # re initialize it
        x_to_fit, y_to_fit = [], []
        for x, y in zip(markers_x, markers_y):
            if abs(y - self.img_height / 2) < coords_th:
                x_to_fit.append(x)
                y_to_fit.append(y)
        x_to_fit = np.array(x_to_fit).reshape(-1, 1)
        y_to_fit = np.array(y_to_fit)
        num_inliers = 0
        while num_inliers != 9:
            # Define RANSAC model
            ransac = linear_model.RANSACRegressor(max_trials=10000, stop_n_inliers=9, residual_threshold=ransac_th_dynamic)

            ransac.fit(x_to_fit, y_to_fit)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            num_inliers = x_to_fit[inlier_mask].shape[0]
            print("Number of inliers", num_inliers)

            if num_inliers == 9:
                print("Found horizontal line!")
                horizontal_line = [ransac.estimator_.coef_[0], ransac.estimator_.intercept_]
                horizontal_points = np.array([x_to_fit[:, 0][inlier_mask], y_to_fit[inlier_mask]]).T

                # Find the central point and remove it from "vertical_points" array
                # OK but does not work if central marker is not actually the closest to center image
                #center_id = self.closest_point(horizontal_points, np.array([self.img_width / 2, self.img_height / 2]))
                # SOlUTION: sort horizontal_points and get the central one
                
                if len(central_point) == 0:
                    columnIndex = 0
                    horizontal_points = horizontal_points[horizontal_points[:, columnIndex].argsort()]
                    center_id = 4
                    central_point = horizontal_points[center_id, :]
                horizontal_points = np.delete(horizontal_points, center_id, 0)
            else:
                ransac_th_dynamic += dyn_increment

        if show_plots:
            print(x_to_fit.shape)
            plt.scatter(x_to_fit[:, 0][inlier_mask], y_to_fit[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
            plt.scatter(x_to_fit[:, 0][outlier_mask], y_to_fit[outlier_mask], color="gold", marker=".", label="Outliers")
            plt.legend(loc="lower right")
            plt.xlabel("u (pixels)")
            plt.ylabel("v (pixels)")
            plt.show()

        # 3 - Find pos/neg diagonal line
        x_to_fit, y_to_fit = [], []
        for x, y in zip(markers_x, markers_y):
            if [x,y] not in vertical_points and [x,y] not in horizontal_points and [x,y] not in central_point:
                x_to_fit.append(x)
                y_to_fit.append(y)
        x_to_fit = np.array(x_to_fit).reshape(-1, 1)
        y_to_fit = np.array(y_to_fit)

        ransac_th_dynamic = ransac_th  # re initialize it
        num_inliers = 0
        while num_inliers != 6:  # because center is not considered
            # Define RANSAC model
            ransac = linear_model.RANSACRegressor(max_trials=10000, stop_n_inliers=9, residual_threshold=ransac_th_dynamic)

            ransac.fit(x_to_fit, y_to_fit)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            num_inliers = x_to_fit[inlier_mask].shape[0]
            print("Number of inliers", num_inliers)

            if num_inliers == 6:
                if ransac.estimator_.coef_[0] > 0:
                    print("Found diagonal positive line!")
                    diag_pos_line = [ransac.estimator_.coef_[0], ransac.estimator_.intercept_]
                    diag_pos_points = np.array([x_to_fit[:, 0][inlier_mask], y_to_fit[inlier_mask]]).T
                else:
                    print("Found diagonal negative line!")
                    diag_neg_line = [ransac.estimator_.coef_[0], ransac.estimator_.intercept_]
                    diag_neg_points = np.array([x_to_fit[:, 0][inlier_mask], y_to_fit[inlier_mask]]).T
            else:
                ransac_th_dynamic += dyn_increment


        if show_plots:
            plt.scatter(x_to_fit[inlier_mask], y_to_fit[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
            plt.scatter(x_to_fit[outlier_mask], y_to_fit[outlier_mask], color="gold", marker=".", label="Outliers")
            plt.legend(loc="lower right")
            plt.xlabel("u (pixels)")
            plt.ylabel("v (pixels)")
            plt.show()

        # 4 - Find neg/pos diagonal line
        x_to_fit, y_to_fit = [], []
        for x, y in zip(markers_x, markers_y):
            if len(diag_pos_points) > 0:
                if [x, y] not in vertical_points and [x, y] not in horizontal_points and \
                [x, y] not in diag_pos_points and [x, y] not in central_point:
                    x_to_fit.append(x)
                    y_to_fit.append(y)
            elif len(diag_neg_points) > 0:
                if [x, y] not in vertical_points and [x, y] not in horizontal_points and \
                [x, y] not in diag_neg_points and [x, y] not in central_point:
                    x_to_fit.append(x)
                    y_to_fit.append(y)
        x_to_fit = np.array(x_to_fit).reshape(-1, 1)
        y_to_fit = np.array(y_to_fit)

        ransac_th_dynamic = ransac_th  # re initialize it
        num_inliers = 0
        while num_inliers != 6:  # because center is not considered
            # Define RANSAC model
            ransac = linear_model.RANSACRegressor(max_trials=10000, stop_n_inliers=9, residual_threshold=ransac_th_dynamic)

            ransac.fit(x_to_fit, y_to_fit)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            num_inliers = x_to_fit[inlier_mask].shape[0]
            print("Number of inliers", num_inliers)

            if num_inliers == 6:
                if ransac.estimator_.coef_[0] > 0:
                    print("Found diagonal positive line!")
                    diag_pos_line = [ransac.estimator_.coef_[0], ransac.estimator_.intercept_]
                    diag_pos_points = np.array([x_to_fit[:, 0][inlier_mask], y_to_fit[inlier_mask]]).T
                else:
                    print("Found diagonal negative line!")
                    diag_neg_line = [ransac.estimator_.coef_[0], ransac.estimator_.intercept_]
                    diag_neg_points = np.array([x_to_fit[:, 0][inlier_mask], y_to_fit[inlier_mask]]).T
            else:
                ransac_th_dynamic += dyn_increment

        if show_plots:
            plt.scatter(x_to_fit[inlier_mask], y_to_fit[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
            plt.scatter(x_to_fit[outlier_mask], y_to_fit[outlier_mask], color="gold", marker=".", label="Outliers")
            plt.legend(loc="lower right")
            plt.xlabel("u (pixels)")
            plt.ylabel("v (pixels)")
            plt.show()

        return vertical_line, horizontal_line, diag_neg_line, diag_pos_line, \
            central_point, vertical_points, horizontal_points, diag_neg_points, diag_pos_points

    '''
    def sort_markers_trajectory(self, all_markers_trajectory, central_point, vert_points, hor_points, diag_neg_points, diag_pos_points, show_plots):
        sorted_trajs = []


        init_coords = all_markers_trajectory[:, 0:2]

        # Append central marker as element 0
        central_marker_id = self.closest_point(init_coords, central_point)
        sorted_trajs.append(all_markers_trajectory[central_marker_id, :])

        # Append sorted vertical points
        columnIndex = 1
        vert_points = vert_points[vert_points[:, columnIndex].argsort()]
        for k in range(8):
            sorted_trajs.append(vert_points[k, :])

        # Append sorted horizontal points
        columnIndex = 0
        hor_points = hor_points[hor_points[:, columnIndex].argsort()]
        for k in range(8):
            sorted_trajs.append(hor_points[k, :])

        # Append diag_pos_points
        columnIndex = 0
        diag_pos_points = diag_pos_points[diag_pos_points[:, columnIndex].argsort()]
        for k in range(6):
            sorted_trajs.append(diag_pos_points[k, :])

        # Append diag_neg_points
        columnIndex = 0
        diag_neg_points = diag_neg_points[diag_neg_points[:, columnIndex].argsort()]
        for k in range(6):
            sorted_trajs.append(diag_neg_points[k, :])
        


        for p in enumerate(vert_points):
            p_x, p_y = p
            for m_id, p_init in enumerate(init_coords):
                init_x, init_y = p_init
                if p_y == init_y and p_x == init_x:
                    sorted_trajs.append(all_markers_trajectory[m_id, :])
        print("vert", np.array(sorted_trajs).shape)

        print("Raw h", hor_points)
        columnIndex = 0
        hor_points = hor_points[hor_points[:, columnIndex].argsort()]
        print("Sorted h", hor_points)

        for p in hor_points:
            p_x, p_y = p
            for m_id, p_init in enumerate(init_coords):
                init_x, init_y = p_init
                if p_x == init_x and p_y == init_y:
                    sorted_trajs.append(all_markers_trajectory[m_id, :])
        print("Final shape", np.array(sorted_trajs).shape)

        print("Raw dp ", diag_pos_points)
        columnIndex = 0
        diag_pos_points = diag_pos_points[diag_pos_points[:, columnIndex].argsort()]
        print("Sorted dp", diag_pos_points)
        for p in diag_pos_points:
            p_x, p_y = p
            for m_id, p_init in enumerate(init_coords):
                init_x, init_y = p_init
                if p_x == init_x and p_y == init_y:
                    sorted_trajs.append(all_markers_trajectory[m_id, :])
        print("Final shape", np.array(sorted_trajs).shape)

        print("Raw dn ", diag_neg_points)
        columnIndex = 0
        diag_neg_points = diag_neg_points[diag_neg_points[:, columnIndex].argsort()]
        print("Sorted dn", diag_neg_points)
        for p in diag_neg_points:
            p_x, p_y = p
            for m_id, p_init in enumerate(init_coords):
                init_x, init_y = p_init
                if p_x == init_x and p_y == init_y:
                    sorted_trajs.append(all_markers_trajectory[m_id, :])
        
        sorted_trajs = np.array(sorted_trajs)
        print("Final shape", np.array(sorted_trajs).shape)

        # Reshape correctly, ex: (53, 29=num.markers, 3)
        sorted_trajs = np.moveaxis(sorted_trajs, 1, 0)

        print("Final shape", np.array(sorted_trajs).shape)

        # Show sorted initial markers
        show_plots = True
        if show_plots:
            plt.scatter(sorted_trajs[:,0], sorted_trajs[:,1], color="yellowgreen", marker=".", label="Markers")
            for m_id in range(sorted_trajs.shape[1]):
                plt.text(sorted_trajs[m_id,0], sorted_trajs[m_id,1], str(m_id), color="black")
            plt.legend(loc="lower right")
            plt.xlabel("u (pixels)")
            plt.ylabel("v (pixels)")
            plt.show()

        return sorted_trajs
    '''   
    def sort_markers_trajectory(self, all_markers_trajectory, central_point, vert_points, hor_points, diag_neg_points, diag_pos_points, show_plots):

        sorted_trajs = []

        init_coords = all_markers_trajectory[0, :, :][:, 0:2]

        # Find the trajectory of the central marker
        central_marker_id = self.closest_point(init_coords, central_point)
        sorted_trajs.append(all_markers_trajectory[-1, central_marker_id, :])

        print("sorted_trajs", np.array(sorted_trajs).shape)

        columnIndex = 1
        vert_points = vert_points[vert_points[:, columnIndex].argsort()]

        for k,p in enumerate(vert_points):
            p_x, p_y = p
            for m_id, p_init in enumerate(init_coords):
                init_x, init_y = p_init
                if p_y == init_y and p_x == init_x:
                    sorted_trajs.append(all_markers_trajectory[-1, m_id, :])
        print("sorted_trajs", np.array(sorted_trajs).shape)


        columnIndex = 0
        hor_points = hor_points[hor_points[:, columnIndex].argsort()]
        for p in hor_points:
            p_x, p_y = p
            for m_id, p_init in enumerate(init_coords):
                init_x, init_y = p_init
                if p_x == init_x and p_y == init_y:
                    sorted_trajs.append(all_markers_trajectory[-1, m_id, :])
        print("sorted_trajs", np.array(sorted_trajs).shape)

        columnIndex = 0
        diag_pos_points = diag_pos_points[diag_pos_points[:, columnIndex].argsort()]
        for p in diag_pos_points:
            p_x, p_y = p
            for m_id, p_init in enumerate(init_coords):
                init_x, init_y = p_init
                if p_x == init_x and p_y == init_y:
                    sorted_trajs.append(all_markers_trajectory[-1, m_id, :])
        print("sorted_trajs", np.array(sorted_trajs).shape)

        columnIndex = 0
        diag_neg_points = diag_neg_points[diag_neg_points[:, columnIndex].argsort()]
        for p in diag_neg_points:
            p_x, p_y = p
            for m_id, p_init in enumerate(init_coords):
                init_x, init_y = p_init
                if p_x == init_x and p_y == init_y:
                    sorted_trajs.append(all_markers_trajectory[-1, m_id, :])

        sorted_trajs = np.array(sorted_trajs)
        print("Final sorted_trajs", sorted_trajs.shape)

        # (29=num.markers, 3)
        #sorted_trajs = np.moveaxis(sorted_trajs, 1, 0) # not required here

        # Show sorted initial markers
        if show_plots:
            plt.scatter(sorted_trajs[:,0], sorted_trajs[:,1], color="yellowgreen", marker=".", label="Markers")
            for m_id in range(sorted_trajs.shape[1]):
                plt.text(sorted_trajs[m_id,0], sorted_trajs[m_id,1], str(m_id), color="black")
            plt.legend(loc="lower right")
            plt.xlabel("u (pixels)")
            plt.ylabel("v (pixels)")
            plt.show()

        return sorted_trajs

                
           
            

    # END press_callback()

# START main()
if __name__ == '__main__':

    force_estimator_node = ForceEstimator()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("ROSInterruptException!")





