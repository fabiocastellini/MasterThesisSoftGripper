#!/usr/bin/env python

import bagpy
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
import random
import shutil


global figure_edges
figure_edges = []

def onclick_trim_edges(event):
    ix, iy = event.xdata, event.ydata
    figure_edges.append(ix)
    if len(figure_edges) == 2:
        fig1.canvas.mpl_disconnect(cid)  # disconnect clicks acquisition
        plt.close(fig1)  # close figure 1

global figure_alignment_times
figure_alignment_times = []
def onclick_alignment_times(event):
    ix, iy = event.xdata, event.ydata
    figure_alignment_times.append(ix)

    if len(figure_alignment_times) == 2:
        fig2.canvas.mpl_disconnect(cid)  # disconnect clicks acquisition
        plt.close(fig2)  # close figure 2

# Function to find nearest element + its index to a "value" inside an "array"
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

# Function to normalize array in the range [t_min, t_max]
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)

# Directory in which there are subfolders with .json and .bag data to be synchronized and cleaned
dir_to_check = os.path.abspath(os.path.dirname(__file__) + "/data_to_synchronize")
print("Checking directory", dir_to_check)


data_counter = 1
for full_dir_path, dirs, files in os.walk(dir_to_check):
    for file in files:
        if file.endswith(".bag"):
            print("--------------------------")
            print("["+str(data_counter)+"] Found bag:", file)
            print("["+str(data_counter)+"] Bag is in dir:", full_dir_path)

            folder_name = os.path.basename(os.path.normpath(full_dir_path))

            # Create a directory containing cleaned and synchronized data for each test
            if not os.path.exists(dir_to_check + "/" + folder_name + "_synchronized"):
                os.mkdir(dir_to_check + "/" + folder_name + "_synchronized")
                os.mkdir(dir_to_check + "/" + folder_name + "_synchronized/synchronized_rgb")

            full_bag_path = os.path.join(full_dir_path, file)

            # Read ROSbag
            b = bagreader(full_bag_path)
            #print(b.topic_table)  # to check bag data
            data = b.message_by_topic(topic='/netft_data')  #/netft_data is the ATI force/torques ROS topic inside the bag
            print("["+str(data_counter)+"] Data extracted from ROSbag is stored at:", data)

            num_start_images_to_cut, num_end_images_to_cut = 0, 0  # define the number of images to cut at the start and at the end

            # Read json data (marker tracking)
            json_files = glob.glob(full_dir_path+"/out_json/*.json")
            if len(json_files) != 1:
                print("No file json or more than one are present in "+ full_dir_path+"/out_json/", file=sys.stderr)
            else:



                df_json = pd.read_json(json_files[0])
                timestamp = df_json['timestamp']; timestamp = np.array(timestamp[0])
                all_markers_traj_cm = df_json["all_markers_traj_cm"]; all_markers_traj_cm = np.array(all_markers_traj_cm[0])
                all_markers_traj_px = df_json["all_markers_traj_px"]; all_markers_traj_px = np.array(all_markers_traj_px[0])

                print(all_markers_traj_px.shape)

                print("["+str(data_counter)+"] Plotting bag info + pixel tracking info...")
                df_rosbag = pd.read_csv(data)
                rosbag_Time = df_rosbag['Time']
                rosbag_force_x = df_rosbag['wrench.force.x']
                rosbag_force_y = df_rosbag['wrench.force.y']
                rosbag_force_z = df_rosbag['wrench.force.z']

                # Remove any bias in the force measurements
                rosbag_force_x = rosbag_force_x - rosbag_force_x[0]
                rosbag_force_y = rosbag_force_y - rosbag_force_y[0]
                rosbag_force_z = rosbag_force_z - rosbag_force_z[0]

                # Important step: normalizing times!
                timestamp_norm = normalize(timestamp, t_min=0, t_max=1)
                rosbag_Time_norm = normalize(rosbag_Time, t_min=0, t_max=1)

                # Plot the rosbag forces and all markers trajectory in pixel units
                print("["+str(data_counter)+"] [Step 1] Click at the start and end of one of the plots in order to trim the edges")
                fig1, ax1 = bagpy.create_fig(3)
                fig1.suptitle('Click on the left and then right to trim the edges', fontsize=16)

                figure_edges = []  # initialize peaks
                cid = fig1.canvas.mpl_connect('button_press_event', onclick_trim_edges)
                ax1[0].scatter(x=rosbag_Time_norm, y=rosbag_force_x, color=(1, 0, 0), s=2, label='wrench.force.x')
                ax1[0].scatter(x=rosbag_Time_norm, y=rosbag_force_y, color=(0, 1, 0), s=2, label='wrench.force.y')
                ax1[0].scatter(x=rosbag_Time_norm, y=rosbag_force_z, color=(0, 0, 1), s=2, label='wrench.force.z')

                num_markers = all_markers_traj_px[:,:,0].shape[0]

                print("[Warning] The number of images MUST be the same of the number of samples (all_markers_traj_px[:,:,0].shape[1])")
                num_total_images = all_markers_traj_px[:,:,0].shape[1]

                for i in range(num_markers):
                    rnd_color = (random.random(), random.random(), random.random())
                    if i == 0:
                        ax1[1].scatter(x=timestamp_norm, y=all_markers_traj_px[i,:,0], color=rnd_color, s=1, label='ALL u-coord [px]')
                        ax1[2].scatter(x=timestamp_norm, y=all_markers_traj_px[i,:,1], color=rnd_color, s=1, label='ALL v-coord [px]')
                    else:
                        ax1[1].scatter(x=timestamp_norm, y=all_markers_traj_px[i, :, 0], color=rnd_color, s=1)
                        ax1[2].scatter(x=timestamp_norm, y=all_markers_traj_px[i, :, 1], color=rnd_color, s=1)
                    ax1[1].plot(timestamp_norm, all_markers_traj_px[i, :, 0], color=rnd_color)
                    ax1[2].plot(timestamp_norm, all_markers_traj_px[i, :, 1], color=rnd_color)

                for axis1 in ax1:
                    axis1.legend()
                    axis1.set_xlabel('Time')
                plt.show()

                print("["+str(data_counter)+"] [Step 1] Start and end normalized time coordinates:", figure_edges)
                _, start_rosbag_id = find_nearest(rosbag_Time_norm, figure_edges[0])
                _, end_rosbag_id = find_nearest(rosbag_Time_norm, figure_edges[1])
                _, start_pixel_id = find_nearest(timestamp_norm, figure_edges[0])
                _, end_pixel_id = find_nearest(timestamp_norm, figure_edges[1])
                num_start_images_to_cut += start_pixel_id
                num_end_images_to_cut = num_total_images - end_pixel_id

                rosbag_force_x = rosbag_force_x[start_rosbag_id:end_rosbag_id]
                rosbag_force_y = rosbag_force_y[start_rosbag_id:end_rosbag_id]
                rosbag_force_z = rosbag_force_z[start_rosbag_id:end_rosbag_id]
                rosbag_Time_norm = normalize(rosbag_Time_norm[start_rosbag_id:end_rosbag_id], t_min=0, t_max=1)  # re-normalize
                all_markers_traj_px = all_markers_traj_px[:, start_pixel_id:end_pixel_id, :]
                timestamp_norm = normalize(rosbag_Time_norm[start_pixel_id:end_pixel_id], t_min=0, t_max=1)  # re-normalize

                print("["+str(data_counter)+"] [Step 2] Click ABOVE and then UNDER to choose the two corresponding peaks")
                fig2, ax2 = bagpy.create_fig(3)
                fig2.suptitle('Click ABOVE and then UNDER to choose the two corresponding peaks', fontsize=16)

                figure_alignment_times = []  # initialize peaks
                cid = fig2.canvas.mpl_connect('button_press_event', onclick_alignment_times)
                ax2[0].scatter(x=rosbag_Time_norm, y=rosbag_force_x, color=(1, 0, 0), s=2, label='wrench.force.x')
                ax2[0].scatter(x=rosbag_Time_norm, y=rosbag_force_y, color=(0, 1, 0), s=2, label='wrench.force.y')
                ax2[0].scatter(x=rosbag_Time_norm, y=rosbag_force_z, color=(0, 0, 1), s=2, label='wrench.force.z')

                num_markers = all_markers_traj_cm[:,:,0].shape[0]
                for i in range(num_markers):
                    rnd_color = (random.random(), random.random(), random.random())
                    if i == 0:
                        ax2[1].scatter(x=timestamp_norm, y=all_markers_traj_px[i,:,0], color=rnd_color, s=1, label='ALL u-coord [px]')
                        ax2[2].scatter(x=timestamp_norm, y=all_markers_traj_px[i,:,1], color=rnd_color, s=1, label='ALL v-coord [px]')
                    else:
                        ax2[1].scatter(x=timestamp_norm, y=all_markers_traj_px[i, :, 0], color=rnd_color, s=1)
                        ax2[2].scatter(x=timestamp_norm, y=all_markers_traj_px[i, :, 1], color=rnd_color, s=1)
                    ax2[1].plot(timestamp_norm, all_markers_traj_px[i, :, 0], color=rnd_color)
                    ax2[2].plot(timestamp_norm, all_markers_traj_px[i, :, 1], color=rnd_color)

                for axis2 in ax2:
                    axis2.legend()
                    axis2.set_xlabel('Time')
                plt.show()

                # Figures are now automatically closed
                print("["+str(data_counter)+"] [Step 2] Forces plots Time-coordinate:", figure_alignment_times[0])
                print("["+str(data_counter)+"] [Step 2] Pixels plots timestamp-coordinate:", figure_alignment_times[1])

                start_rosbag_time, start_rosbag_id = find_nearest(rosbag_Time_norm, figure_alignment_times[0])
                #print("start_rosbag_id" , start_rosbag_time, start_rosbag_id)
                start_pixel_time, start_pixel_id = find_nearest(timestamp_norm, figure_alignment_times[1])
                #print("start_pixel_id", start_pixel_time, start_pixel_id)

                # Shift the json pixel "timestamp" and trim the edges, then re-normalize in [0,1] range
                delta_t = figure_alignment_times[0]-figure_alignment_times[1] # force - pixel
                #print("delta_t", delta_t)
                if delta_t < 0:  # need to cut pixels "signal"
                    _, num_samples_to_cut = find_nearest(timestamp_norm, abs(delta_t))
                    num_start_images_to_cut += num_samples_to_cut #only cut at the start
                    timestamp_norm = timestamp_norm[num_samples_to_cut:]
                    all_markers_traj_px = all_markers_traj_px[:, num_samples_to_cut:, :]
                    #print("num_samples_to_cut TIMESTAMP", num_samples_to_cut)

                    _, num_samples_to_cut = find_nearest(rosbag_Time_norm, abs(delta_t))
                    #print("num_samples_to_cut ROS", num_samples_to_cut)

                    tot_samples_rosbag = len(rosbag_Time_norm)
                    rosbag_force_x = rosbag_force_x[:(tot_samples_rosbag - num_samples_to_cut)]
                    rosbag_force_y = rosbag_force_y[:(tot_samples_rosbag - num_samples_to_cut)]
                    rosbag_force_z = rosbag_force_z[:(tot_samples_rosbag - num_samples_to_cut)]
                    rosbag_Time_norm = rosbag_Time_norm[:(tot_samples_rosbag - num_samples_to_cut)]

                    timestamp_norm = timestamp_norm + delta_t  # update the json timestamp!
                else:
                    print("Error: signal1 timestamp has to be less than signal2 timestamp, reverse them!",file=sys.stderr)

                timestamp_norm = normalize(timestamp_norm, t_min=0, t_max=1)  # re-normalize
                rosbag_Time_norm = normalize(rosbag_Time_norm, t_min=0, t_max=1)  # re-normalize

                # -------------------------------------------
                # Same code of above to visualize the alignment results.......if needed make it better:
                # -------------------------------------------
                # Plot the rosbag forces and all markers trajectory in pixel and metric units
                fig2, ax2 = bagpy.create_fig(3)
                #print("rosbag_Time_norm.shape, rosbag_force_x.shape", rosbag_Time_norm.shape, rosbag_force_x.shape)
                #print("timestamp_norm.shape, all_markers_traj_px.shape", timestamp_norm.shape, all_markers_traj_px.shape)

                ax2[0].scatter(x=rosbag_Time_norm, y=rosbag_force_x, color=(1, 0, 0), s=2, label='wrench.force.x')
                ax2[0].scatter(x=rosbag_Time_norm, y=rosbag_force_y, color=(0, 1, 0), s=2, label='wrench.force.y')
                ax2[0].scatter(x=rosbag_Time_norm, y=rosbag_force_z, color=(0, 0, 1), s=2,  label='wrench.force.z')

                num_markers = all_markers_traj_px[:, :, 0].shape[0]
                for i in range(num_markers):
                    rnd_color = (random.random(), random.random(), random.random())
                    if i == 0:
                        ax2[1].scatter(x=timestamp_norm, y=all_markers_traj_px[i, :, 0], color=rnd_color, s=1, label='ALL u-coord [px]')
                        ax2[2].scatter(x=timestamp_norm, y=all_markers_traj_px[i, :, 1], color=rnd_color, s=1, label='ALL v-coord [px]')
                    else:
                        ax2[1].scatter(x=timestamp_norm, y=all_markers_traj_px[i, :, 0], color=rnd_color, s=1)
                        ax2[2].scatter(x=timestamp_norm, y=all_markers_traj_px[i, :, 1], color=rnd_color, s=1)
                    ax2[1].plot(timestamp_norm, all_markers_traj_px[i, :, 0], color=rnd_color)
                    ax2[2].plot(timestamp_norm, all_markers_traj_px[i, :, 1], color=rnd_color)

                for axis2 in ax2:
                    axis2.legend()
                    axis2.set_xlabel('Time')
                plt.show()
                #-------------------------------------------

                # Writing synchronized signals to json files
                with open(dir_to_check + "/" + folder_name + "_synchronized/synchronized_pixel.json", "w") as outfile:
                    json_data = [np.array(timestamp_norm), np.array(all_markers_traj_px)]
                    df_json = pd.DataFrame([json_data], index=None, columns=["timestamp_norm", "all_markers_traj_px"])
                    print("Writing results...\n", df_json.head(), "\n", df_json.info)
                    json_object = df_json.to_json()
                    outfile.write(json_object)

                with open(dir_to_check + "/" + folder_name + "_synchronized/synchronized_rosbag.json", "w") as outfile:
                    json_data = [np.array(rosbag_Time_norm), np.array(rosbag_force_x), np.array(rosbag_force_y), np.array(rosbag_force_z)]
                    df_json = pd.DataFrame([json_data], index=None, columns=["rosbag_Time_norm", "rosbag_force_x", "rosbag_force_y", "rosbag_force_z"])
                    print("Writing results...\n", df_json.head(), "\n", df_json.info)
                    json_object = df_json.to_json()
                    outfile.write(json_object)

                print("Copying the synchronized images in the range", num_start_images_to_cut,"-",  (num_total_images-num_end_images_to_cut))
                for f_d_p, ds, fs in os.walk(full_dir_path + "/out_img/rgb"):
                    for f in fs:
                        # all images MUST be saved in the format {N_tracking}json_{N_frame}frame, ex: "1json_2frame"
                        if f.endswith(".png"):
                            second_name_part = f.split("_")[1]
                            frame_number = int(second_name_part.split("frame")[0])
                            # TODO Careful: <= frame_number < signes are because frame_number starts from 2...to correct in main logs if needed!
                            if num_start_images_to_cut <= frame_number < (num_total_images - num_end_images_to_cut):
                                destination_path = dir_to_check + "/" + folder_name + "_synchronized/synchronized_rgb"
                                #print("Copying", frame_number, "image to", destination_path)
                                shutil.copyfile(f_d_p+"/"+f, destination_path+"/"+f)

                data_counter += 1

                print("--------------------------\n")
        break



