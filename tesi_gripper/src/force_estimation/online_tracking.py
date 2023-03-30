# Import python libraries
import cv2
import pandas as pd
import numpy as np
import time
import glob
import re
from scipy.signal import butter, filtfilt
import pandas as pd

# Import my libraries
from visualize_results import *
from blob_detection import *

# Low pass filtering function
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normalCutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def track_markers(marker_shape, num_markers_gripper, distance_th, initial_coords):

    # Initialize coords and timestamps lists
    all_traj_markers = []      # initialize all detected markers' list
    all_times = []             # initialize all timestamps' list

    # Initialize matrix that will indicate if interpolation is required for that iteration (row) and marker (col)
    interpolation_matrix = [[False for i in range(num_markers_gripper)]]

    all_traj_markers.append(initial_coords.copy())

    print("[1 - Offline tracking] Initialization done!")
    all_times.append(time.time_ns())

    prev_coords = initial_coords
    for img_path in files:
        # Read input image
        raw_frame = cv2.imread(img_path)
        cv2.imshow("Receiving frame", raw_frame)
        cv2.waitKey(1)

        # Detect markers
        frame_with_markers, new_markers, num_detected_markers = detect_markers(raw_frame.copy(), marker_shape)
        cv2.imshow("Frame with markers", frame_with_markers)  # show frame with markers, useful to know when to press space bar
        cv2.waitKey(1)

        all_times.append(time.time_ns())

        interpolation_vector = []
        traj_update = []

        filter_cutoff_freq = 5  # lpf params
        filter_order = 2        # lpf params
        Ts = 0.024 #(all_times[-1]-all_times[-2])/1000000000/2
        #print(Ts) # to tune the value and avoid recomputing it every time / changing the filtering output
        fs = 1 / Ts  # frequency
        # print("Pixels sample rate [Hz]:", fs)

        for marker_id in range(num_markers_gripper):
            # compute distances between old center and all new markers to see what's the closest
            old_center = np.array((prev_coords[marker_id][0], prev_coords[marker_id][1]))

            eucl_dists = []
            for m in new_markers:
                new_center = np.array((m[0], m[1]))
                eucl_dists.append(np.linalg.norm(old_center - new_center))

            if min(eucl_dists) < distance_th:  # impose threshold (else means marker was not detected in actual frame)
                argmin_id = np.argmin(np.array(eucl_dists))

                traj_update.append(new_markers[argmin_id])
                interpolation_vector.append(False)
                #print(new_markers[argmin_id], "is close to", prev_coords[marker_id])
            else:
                traj_update.append(prev_coords[marker_id])  # append the old values (already filtered)
                interpolation_vector.append(True)   # remember that interpolation should be done later
                #print("Marker is FAR from others, adding", prev_coords[marker_id])

        all_traj_markers.append(traj_update)
        interpolation_matrix.append(interpolation_vector)
        prev_coords = traj_update

        all_traj_markers_arr = np.array(all_traj_markers)

        # Low pass filtering of pixel coordinates
        if all_traj_markers_arr.shape[0] > 10:
            for m_id in range(num_markers_gripper):
                for coord_id in range(3):
                    all_traj_markers_arr[:,m_id,coord_id] = butter_lowpass_filter(all_traj_markers_arr[:,m_id,coord_id], cutoff = filter_cutoff_freq, fs = fs, order = filter_order)
        all_traj_markers = list(all_traj_markers_arr)
        print("------temp----------")
        print(traj_update)
        print("-------------")

    print("[1 - Offline tracking] [End] Offline tracking ended!")
    # ------------------------------------------------------


    print("[1 - Offline tracking] Shape all_traj_markers:", np.array(all_traj_markers).shape)
    print("[1 - Offline tracking] Shape interpolation_matrix:", np.array(interpolation_matrix).shape)

    # Do linear interpolation where interpolation matrix value is True
    for iteration, interpolation_arr in enumerate(interpolation_matrix):
        interpolation_arr = np.array(interpolation_arr)
        ind = list(np.where(interpolation_arr == True)[0])

        if len(ind) > 0:
            for marker_id in ind:
                print(all_traj_markers[iteration][marker_id])
                all_traj_markers[iteration][marker_id] = [np.nan, np.nan, np.nan]
    #print("Before", np.argwhere(np.isnan(all_traj_markers))) # to check where are the np.nan values

    # Interpolation step: if "np.nan", means tracking was lost, so interpolate linearly between positions
    trajs = np.array(all_traj_markers)
    for marker_id in range(num_markers_gripper):
        interpolated_col_df = pd.DataFrame(trajs[:,marker_id,:]).interpolate()

        for k, el in enumerate(interpolated_col_df.values.tolist()):
            all_traj_markers[k][marker_id] = el
    #print("After", np.argwhere(np.isnan(all_traj_markers))) # to check where are the np.nan values

    # Convert timestamps to seconds
    all_times = np.array(all_times)
    all_times = all_times/1000000000 - all_times[0]/1000000000

    # Compute displacements by removing the initial bias to the markers' trajectories
    px_disps = np.array(all_traj_markers)
    print("[1 - Offline tracking] Shape pixel displacements:", px_disps.shape)
    px_disps = px_disps - px_disps[0,:,:]

    print("[1 - Offline tracking] Saving output json file to:", out_json_path)

    # Create pandas dataframe from list "" and convert to "json_object"
    json_data = [np.array(all_times), np.array(all_traj_markers), px_disps]
    df_json = pd.DataFrame([json_data], index=None, columns=["timestamp_seconds", "all_markers_traj_px",  "all_markers_displacements"])
    json_object = df_json.to_json()

    # Writing results to detected_objects.json
    with open(out_json_path, "w") as outfile:
        outfile.write(json_object)

    show_markers_displacements(all_times, px_disps)




"""
def track_markers_motion(all_detected_markers, interpolate):
    max_radius_increment = 0.5  # to be tuned according to the gripper design

    #interpolate = False  # TODO: address interpolation here, not in main
    all_markers_traj = []
    num_markers = len(all_detected_markers[0])

    for marker_id in range(num_markers):
        marker_traj = []
        marker_pos = all_detected_markers[0][marker_id]  # starting position for "marker_id" marker

        marker_traj.append(marker_pos)  # first position of the "marker_id"-th marker

        for k in range(len(all_detected_markers)-1):
            closest = find_closest_marker(marker_pos, all_detected_markers[k+1])
            if not np.isnan(closest[0]): #if a close marker was found
                # check if there's a quick radius increment between prev and actual radius
                prev_radius = marker_pos[2]
                actual_radius = closest[2]
                if quick_radius_increment(prev_radius, actual_radius, max_radius_increment):
                    closest = [np.nan, np.nan, np.nan]
                    interpolate = True
            marker_traj.append(closest)  #append actual closest, or None, then interpolate!
            marker_pos = closest #go on anyway (is it ok?)
        all_markers_traj.append(marker_traj)

    if interpolate:
        # Interpolation step: if "np.nan", means tracking was lost, so interpolate linearly between positions
        #print("before interpolation:", all_markers_traj)  #to debug
        for i, traj in enumerate(all_markers_traj):
            interpolated_traj_df = pd.DataFrame(traj).interpolate()
            interpolated_traj = interpolated_traj_df.values.tolist()

            all_markers_traj[i] = interpolated_traj
        #print("AFTER interpolation:", all_markers_traj)  #to debug
    return all_markers_traj





def quick_radius_increment(prev_radius, actual_radius, max_radius_increment):
    if abs(actual_radius - prev_radius) > max_radius_increment:
        return True
    else:
        return False



def find_closest_marker(old_marker, all_new_markers):
    old_center = np.array((old_marker[0], old_marker[1]))

    eucl_dists = []
    for new_marker in all_new_markers:
        new_center = np.array((new_marker[0], new_marker[1]))
        eucl_dists.append(np.linalg.norm(old_center - new_center))

    # 20 is ok for the 9 markers gripper
    if min(eucl_dists) < 5:  # impose threshold (else means marker was not detected in actual frame)
        argmin_id = np.argmin(np.array(eucl_dists))
        return all_new_markers[argmin_id]
    else:
        return [np.nan, np.nan, np.nan]  #return this list because it's useful to then interpolate with pandas


def pixel_to_meters(initial_px, pixel, cx, cy, fx, Omm, Imm, Ipx):
    u, v, radius_2 = pixel
    _,_, radius_1 = initial_px
    rho_1 = 20.05  #mm radius of half sphere (no sphere depth) without deformation

    Opx = 2 * radius_2  # wrong, because circles become ellipses...
    if abs(Opx) < 0.0001:
        Opx = 0.0001
        radius_2 = 0.0001

    X_hat = (u - cx)*Omm/Opx #np.round((u - cx)*Omm/Opx, 0)
    Y_hat = (cy - v)*Omm/Opx #np.round((cy - v)*Omm/Opx, 0)
    K = 1000  # K = fx*Ipx/Imm
    #print("u-cx:",np.round((u - cx),0))
    #print("cy-v:", np.round((cy - v),0))
    #print("Omm/Opx", Omm/Opx, Omm/Opx*K)
    Z_hat = -1 #np.sqrt(K*Omm/Opx -X_hat**2)
    #print("X_hat, Y_hat, Z_hat", X_hat, Y_hat, Z_hat)


    '''
    rho_hat = rho_1 * radius_1 / radius_2
    if rho_hat > rho_1:  # due to noisy radius measurements, could happen --> must be filtered!
        rho_hat = rho_1
    elif abs(rho_hat) < 0.001:
        rho_hat = 0.001

    if abs(np.sqrt((X_hat**2 + Y_hat**2)/rho_hat**2)) > 1:
        phi_hat = 0 # in this case point will be on rho_hat
    else:
        phi_hat = np.arcsin( np.sqrt((X_hat**2 + Y_hat**2)/rho_hat**2) )

    Z_hat = rho_hat * np.cos(phi_hat)
    '''

    return [X_hat, Y_hat, Z_hat]



def get_markers_traj_cm(all_markers_traj_px, Omm, cx, cy, fx, fy, Imm, Ipx):
    all_markers_traj_cm = []

    for j,traj_px in enumerate(all_markers_traj_px):
        print("all_markers_traj_px num.", j)

        traj_3d = []
        initial_px = traj_px[0]

        for k, pos_px in enumerate(traj_px):
            print("traj_px num.", k)
            point_3d = pixel_to_meters(initial_px, pos_px, cx, cy, fx, Omm, Imm,   Ipx)
            traj_3d.append(point_3d)
        all_markers_traj_cm.append(traj_3d)

    initial_coords_cm = np.array(all_markers_traj_cm)[:,0,:]  #get initial coordinates in cm
    return all_markers_traj_cm, initial_coords_cm


def get_markers_displacements(initial_coords, all_markers_traj):
    initial_coords = np.array(initial_coords)
    all_markers_traj = np.array(all_markers_traj)

    '''
    displacements = []  # list of "num_markers" lists containing x,y and radius (not meaningful?) displacements during deformation
    for marker_id, traj in enumerate(all_markers_traj):
        disp = np.array([0,0,0])
        for coords in traj:
            consecutive_disp = np.subtract( coords, initial_coords[marker_id,:])
            disp = np.sum([disp, consecutive_disp], axis=0)
        #abs_disp = np.absolute(disp) # why??? 2-12 correction
        #displacements.append(abs_disp)
        displacements.append(disp)
    '''

    # Try another approach: consecutive displacemetnts can sum up measurements errors
    displacements = []  # list of "num_markers" lists containing x,y and radius (not meaningful?) displacements during deformation
    for marker_id in range(len(all_markers_traj)):
        displacements.append( np.subtract(all_markers_traj[marker_id,-1,:], initial_coords[marker_id, :]) )

    displacements = np.array(displacements)
    return displacements



# Force application point is wrong and not so easy to find this way...
def find_force_application_point(all_markers_traj_cm):

    #print("displacements", displacements)    #to debug
    #print("initial_coords", initial_coords)  #to debug
    displacements = get_markers_displacements(all_markers_traj_cm)

    all_markers_traj_cm = np.array(all_markers_traj_cm)
    initial_coords = all_markers_traj_cm[:, 0, :]
    if np.sum(displacements[:, 0]) == 0:
        x_f = 0
    else:
        #x_f = np.dot(initial_coords[:, 0], displacements[:, 0]) / np.sum(displacements[:, 0])
        x_f = np.dot(initial_coords[:, 0],  displacements[:, 2]) / np.sum(displacements[:, 2])
    if np.sum(displacements[:, 1]) == 0:
        y_f = 0
    else:
        #y_f = np.dot(initial_coords[:,1] , displacements[:, 1]) / np.sum(displacements[:, 1])
        y_f = np.dot(initial_coords[:,1] , displacements[:, 2]) / np.sum(displacements[:, 2])

    z_f = 22  # mm

    print("Estimated point of application in [mm]:", x_f, y_f)
    return x_f, y_f, z_f, displacements



def check_if_interpolation_needed(all_detected_markers):
    all_valid_readings = []
    for k, m in enumerate(all_detected_markers):
        valid_readings = 0
        for coords in m:
            if not np.isnan(coords[0]):
                valid_readings += 1
        # To debug and see the number of readings and values for each detected marker:
        # print("detection" + str(k) + " - " + str(valid_readings) + " different reading (1 for every marker)")
        # print("Markers' readings (" + str(valid_readings) + ")", m)
        all_valid_readings.append(valid_readings)

    # Check if every marker was detected the same number of time
    if all(el == all_valid_readings[0] for el in all_valid_readings):  # yes --> interpolation not required
        to_interpolate = False
    else:  # no --> interpolation required!
        to_interpolate = True
    return to_interpolate
    
    
    
"""