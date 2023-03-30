# ----------------------------------------------
# This script contains camera calibration data
# ----------------------------------------------
import numpy as np

camera_size = (320, 240)

# Data collected during calibration 23-11-22:
camera_matrix = np.array([[135.51812997, 0., 144.29961957], [0., 134.98688773, 117.63284229], [0., 0., 1.]])
distortion_coefficients = np.array([[-0.29341196, 0.13089203, - 0.00413831, 0.00733319, - 0.02926107]])

# Reminder (12-12-22): cx and cy should be at the frame center, meaning 320/2 and 240/2, but according to camera calibration
#                      resulted 144.29, 117.63. Meters estimation looks better with 160, 120...more tests to be made.
cx, cy = 160, 120 #camera_matrix[0,2], camera_matrix[1,2]
fx, fy = camera_matrix[0,0], camera_matrix[1,1]

# source: https: // www.arducam.com / docs / cameras - for -raspberry - pi / native - raspberry - pi - cameras / 5mp - ov5647 - standard - camera - modules /  # image-sensor
#Imm = 63.5  # 1/4'' = 0.635cm = 63.5mm
#Ipx = 2592  # pixels: 2592×1944

#source: https://www.arducam.com/downloads/modules/RaspberryPi_camera/OV5647DS.pdf
Ipx = 320     # pixel width after crop
Imm = 3.6736/2  # image sensor metric units: 3673.6 µm x 2738.4 µm
