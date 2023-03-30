# Import Python libraries
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



# Import my libraries
from colors import *

def view_marker_trajectory(all_markers_traj, frame):
    print("View marker trajectories...")
    overlay = frame.copy()

    #cv2.circle(overlay, (int(xf_px), int(yf_px)), 3, (0,0,255), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    for marker_id, marker_traj in enumerate(all_markers_traj):
        put_text = False
        for circle in marker_traj:
            x, y, radius = circle
            circle_color = set_of_colors[marker_id]

            if not put_text:
                org = (int(x), int(y))
                #overlay = cv2.putText(overlay, str(marker_id), org, font, fontScale, circle_color, thickness, cv2.LINE_AA)
                put_text = True
            cv2.circle(overlay, (int(x), int(y)), int(radius), circle_color, 1)
            cv2.circle(overlay, (int(x), int(y)), 1, circle_color, 1)
            cv2.line(overlay, (int(x) - 5, int(y)), (int(x) + 5, int(y)), circle_color, 1)
            cv2.line(overlay, (int(x), int(y) - 5), (int(x), int(y) + 5), circle_color, 1)
        opacity = 0.5
        cv2.addWeighted(overlay, opacity, frame, 1-opacity, 0, frame)
        cv2.imshow("trajectory", frame)
        cv2.waitKey(1)



def view_marker_disp_vectors(all_markers_traj, frame):
    print("View marker displacement vectors...")
    overlay = frame.copy()

    #cv2.circle(overlay, (int(xf_px), int(yf_px)), 3, (0,0,255), 3)

    thickness = 2
    for marker_id, marker_traj in enumerate(all_markers_traj):
        start_point = marker_traj[0][0:2]
        max_disp = -1
        visual_factor = 8  # increment arrow length to make it more visible
        for circle in marker_traj:
            x, y, radius = circle
            circle_color = set_of_colors[marker_id]

            actual_disp = np.linalg.norm(np.array(start_point) - np.array([x, y]))

            if max_disp < abs(actual_disp):
                max_disp = abs(actual_disp)
            else:
                start_point = (int(start_point[0]), int(start_point[1]))

                # warning: I incremented the length!
                if visual_factor != 1:
                    if   x > start_point[0] and y > start_point[1]:                  # to right and down
                        end_point = (int(x+visual_factor), int(y+visual_factor))
                    elif x > start_point[0] and y < start_point[1]:                  # to right and up
                        end_point = (int(x+visual_factor), int(y-visual_factor))
                    elif x < start_point[0] and y > start_point[1]:                  # to left and down
                        end_point = (int(x-visual_factor), int(y+visual_factor))
                    elif x < start_point[0] and y < start_point[1]:                  # to left and up
                        end_point = (int(x-visual_factor), int(y-visual_factor))
                else:
                    end_point = (int(x), int(y))

                overlay = cv2.arrowedLine(overlay, start_point, end_point, circle_color, thickness, tipLength=0.5)
                break

        opacity = 0.7
        cv2.addWeighted(overlay, opacity, frame, 1-opacity, 0, frame)
        cv2.imshow("Markers' movements", frame)
        cv2.waitKey(1)


def plot_2d_marker_trajectory_cm(all_markers_traj_cm, xf_cm, yf_cm):
    fig = plt.figure("2d plot [mm]")
    # plt.scatter(xf_cm, yf_cm, c='red')

    all_markers_traj_cm = np.array(all_markers_traj_cm)

    for k, marker_trajectories in enumerate(all_markers_traj_cm):
        marker_trajectories = np.array(marker_trajectories)
        x, y, z = marker_trajectories.T
        cols = []
        temp = set_of_colors_norm[k][0]
        set_of_colors_norm[k][0] = set_of_colors_norm[k][2]
        set_of_colors_norm[k][2] = temp
        for i in range(len(marker_trajectories)):
            cols.append(set_of_colors_norm[k])

        plt.scatter(x, y, c=np.array(cols))
        plt.text(x[0], y[0], "  marker " + str(k), c=np.array(cols[0]))

    plt.grid()
    plt.show()


def plot_3d_initial_markers(all_markers_traj_cm, xf_cm, yf_cm, zf_cm):
    fig = plt.figure("3d plot [mm]")
    ax = plt.axes(projection='3d')

    all_markers_traj_cm = np.array(all_markers_traj_cm)
    print(all_markers_traj_cm.shape)
    all_markers_traj_cm = all_markers_traj_cm[:,0,:]
    for k, marker_trajectories in enumerate(all_markers_traj_cm):
        x, y, z = np.array(marker_trajectories).T

        cols = []
        temp = set_of_colors_norm[k][0]
        set_of_colors_norm[k][0] = set_of_colors_norm[k][2]
        set_of_colors_norm[k][2] = temp
        #for i in range(int(len(x))):
        #    cols.append(list(set_of_colors_norm[k]))
        cols.append(list(set_of_colors_norm[k]))

        ax.scatter3D(x, y, z, c=np.array(cols))
    plt.grid()
    plt.show()



def compare_3d_markers_with_cad(all_markers_traj_cm):
    all_markers_traj_cm = np.array(all_markers_traj_cm)
    X, Y, Z = all_markers_traj_cm[:, :, 0].copy(), all_markers_traj_cm[:, :, 1].copy(), all_markers_traj_cm[:, :, 2].copy()
    all_markers_traj_cm[:, :, 0] = Y
    all_markers_traj_cm[:, :, 1] = Z
    all_markers_traj_cm[:, :, 2] = X

    initial_markers = all_markers_traj_cm[:,0,:]
    final_markers = all_markers_traj_cm[:,-1,:]

    ax = plot_ply('only_markers_modified.ply')
    ax.scatter(initial_markers[:,0], initial_markers[:,1], initial_markers[:,2], c='g', marker='o', s=100)
    ax.scatter(final_markers[:,0], final_markers[:,1], final_markers[:,2], c='b', marker='+', s=100)

    plt.show()

    # Works with Open3D but my points are really small
    """
    # To correct the reference frame:
    all_markers_traj_cm[:, :, 0] = Y
    all_markers_traj_cm[:, :, 1] = Z
    all_markers_traj_cm[:, :, 2] = X
    
    #dome_pcd = o3d.io.read_point_cloud("Dome_full_39_external_holes_v1.ply")  # Read the point cloud
    dome_pcd = o3d.io.read_point_cloud("only_markers.ply")  # Read the point cloud
    dome_pcd.paint_uniform_color([0,0,0])

    initial_markers = all_markers_traj_cm[:,0,:]
    final_markers = all_markers_traj_cm[:,-1,:]

    initial_3d_points_pcd = o3d.geometry.PointCloud()
    initial_3d_points_pcd.points = o3d.utility.Vector3dVector(initial_markers)
    initial_3d_points_pcd.paint_uniform_color([0,0,1]) #initial = blue

    final_3d_points_pcd = o3d.geometry.PointCloud()
    final_3d_points_pcd.points = o3d.utility.Vector3dVector(final_markers)
    final_3d_points_pcd.paint_uniform_color([1,0,0]) #final = red


    #o3d.visualization.draw_geometries([dome_pcd, estimated_3d_points_pcd])  # Visualize the point cloud
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    print(f'Center of mesh: {mesh.get_center()}')

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()

    viewer.add_geometry(dome_pcd)
    viewer.add_geometry(initial_3d_points_pcd)
    viewer.add_geometry(final_3d_points_pcd)
    viewer.add_geometry(mesh)

    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    viewer.run()
    """



def get_pts(infile):
    data = np.loadtxt(infile, delimiter=' ', skiprows=13)
    return data[:,0], data[:,1], data[:,2]  # returns X,Y,Z points skipping the first 12 lines


def plot_ply(infile):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = get_pts(infile)
    ax.scatter(x,y,z, c='r', marker='o')  #account for ref. frame rotation
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return ax


# THESIS: To show the 3 u-v-radius displacements in a graph
def show_markers_displacements(all_times, all_displacements_px):
    # Show markers' displacements u-v-radius
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(all_times, all_displacements_px[:,:,0])
    axs[0].set_ylabel('Pixel displacements [u]')
    axs[0].grid(True)
    axs[0].set_xlabel('Time [s]')

    axs[1].plot(all_times, all_displacements_px[:,:,1])
    axs[1].set_ylabel('Pixel displacements [v]')
    axs[1].grid(True)
    axs[1].set_xlabel('Time [s]')

    axs[2].plot(all_times, all_displacements_px[:,:,2])
    axs[2].set_ylabel('Pixel displacement [radius]')
    axs[2].grid(True)
    axs[2].set_xlabel('Time [s]')

    fig.tight_layout()
    plt.show()



def show_markers_displacements_against_ground_truths(axs, rosbag_Time_norm, rosbag_force_x, rosbag_force_y, rosbag_force_z, timestamp_norm, all_markers_displacements):
    # Show markers' displacements u-v-radius and ground truths
    axs[0].plot(rosbag_Time_norm, rosbag_force_x, color=(1, 0, 0), label='wrench.force.x')
    axs[0].plot(rosbag_Time_norm, rosbag_force_y, color=(0, 1, 0), label='wrench.force.y')
    axs[0].plot(rosbag_Time_norm, rosbag_force_z, color=(0, 0, 1), label='wrench.force.z')
    axs[0].grid(True)
    axs[0].set_ylabel('Ground truth Force components [N] ')
    axs[0].set_xlabel('Time [s]')

    axs[1].plot(timestamp_norm, all_markers_displacements[:,:,0])
    axs[1].set_ylabel('Pixel displacements [u]')
    axs[1].grid(True)
    axs[1].set_xlabel('Time [s]')

    axs[2].plot(timestamp_norm, all_markers_displacements[:,:,1])
    axs[2].set_ylabel('Pixel displacements [v]')
    axs[2].grid(True)
    axs[2].set_xlabel('Time [s]')

    axs[3].plot(timestamp_norm, all_markers_displacements[:,:,2])
    axs[3].set_ylabel('Pixel displacement [radius]')
    axs[3].grid(True)
    axs[3].set_xlabel('Time [s]')

    axs[1].plot(timestamp_norm, np.mean(all_markers_displacements[:, :, 0], axis=1), color=(1, 0, 0), label='Average displacement [u]')
    axs[2].plot(timestamp_norm, np.mean(all_markers_displacements[:, :, 1], axis=1), color=(0, 1, 0), label='Average displacement [v]')
    axs[3].plot(timestamp_norm, np.mean(all_markers_displacements[:, :, 2], axis=1), color=(0, 0, 1), label='Average displacement [radius]')

    plt.show()


