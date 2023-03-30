# Import Python libraries
import socket
import pickle
import struct
import cv2

# Import my libraries
from camera_params import *

def connect_to_server(host_ip, port):
    # Create client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host_ip, port))  # a tuple
    data = b''
    payload_size = struct.calcsize("Q")
    return client_socket, payload_size, data

def read_camera_frame(client_socket, payload_size, data, undistort_frame=False):
    while len(data) < payload_size:
        packet = client_socket.recv(4 * 1024)  # 4K
        if not packet:
            break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]

    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4 * 1024)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame = pickle.loads(frame_data)

    if undistort_frame:
        # Undistort the frame:
        # scaled_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, camera_size, 1, camera_size)
        # roi_x, roi_y, roi_w, roi_h = roi
        undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None, camera_matrix)

        # cropped_frame = undistorted_frame[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
        cv2.imshow("Undistorted frame %s" % (undistorted_frame.shape,), undistorted_frame)
        # cv2.imshow("cropped %s" % (cropped_frame.shape,), cropped_frame)
        frame = undistorted_frame  #consider the undistorted frame

    # Manually define a circle of interest
    raw_frame = frame.copy()

    overlay = frame.copy()
    shape = frame.shape
    center_frame = [int(shape[1] / 2), int(shape[0] / 2)]
    center_radius = int(shape[1] / 2.8)

    # draw the cirlce of interest + draw horizontal and vertical central lines
    cv2.circle(overlay, (center_frame[0], center_frame[1]), center_radius, (0, 255, 0), 5)
    cv2.line(overlay, (50, center_frame[1]), (shape[1] - 50, center_frame[1]), (0, 255, 0), 1)  # vertical
    cv2.line(overlay, (center_frame[0], 50), (center_frame[0], shape[0] - 50), (0, 255, 0), 1)  # horizontal
    opacity = 0.5
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    return raw_frame, frame, data
