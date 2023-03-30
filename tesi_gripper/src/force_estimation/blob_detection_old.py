import cv2
import numpy as np
import random
from scipy.spatial.distance import cdist
import math


# TOCHECK; https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
#https://github.com/pmkalshetti/ellipse_fitting
def set_parameters():
    # Initialize parameters of the SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area
    params.filterByArea = True
    """
    # Gripper with 9 markers
    params.minArea = 100  # suggested before undistortion: 200
    params.maxArea = 900  # suggested before undistortion: 900
    """
    # Gripper with more markers
    # Radial shape (7 lines) = 60 - 500
    # Square shape =
    # Cross shape 29 markers =

    params.minArea = 20
    params.maxArea = 300

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.7

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.5

    #params.minThreshold = 10
    params.maxThreshold = 125

    # Distance Between Blobs
    #params.minDistBetweenBlobs = 0.1


    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    return detector


def detect_markers_ellipses(frame):
    minimum_major_axis = 5  # min. pixel length of ellipse's major axis

    # Convert image to gray and blur it
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #src_gray = cv2.blur(src_gray, (3, 3))
    cv2.imshow("src_gray", src_gray)

    canny_threshold = 120  # initial threshold

    canny_output = cv2.Canny(src_gray, canny_threshold, canny_threshold*2)
    cv2.imshow("canny_output", canny_output)

    #contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(frame[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # Find the rotated rectangles and ellipses for each contour
    minRect = [None] * len(contours)
    minEllipse = [None] * len(contours)

    markers = []
    num_ellipses = 0
    for i, c in enumerate(contours):

        minRect[i] = cv2.minAreaRect(c)
        if c.shape[0] > 5:

            minEllipse[i] = cv2.fitEllipse(c)
            (xc, yc), (d1, d2), angle = minEllipse[i]
            #print("Found ellipse [xc, yc, d1, d2, angle]:\n", xc, yc, d1, d2, angle)

            # Draw contours + rotated rects + ellipses (draw vertical line; compute major radius)
            rmajor = max(d1, d2) / 2
            if angle > 90:
                angle = angle - 90
            else:
                angle = angle + 90
            xtop = xc + math.cos(math.radians(angle)) * rmajor
            ytop = yc + math.sin(math.radians(angle)) * rmajor
            xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
            ybot = yc + math.sin(math.radians(angle + 180)) * rmajor
            major_axis = np.sqrt((xtop - xbot) ** 2) + ((ytop - ybot) ** 2)
            #print(xtop, ytop, xbot, ybot, "dist:", major_axis)

            # -------------------
            # Store marker:
            center = [xc, yc]
            shape = frame.shape
            center_frame = [int(shape[1] / 2), int(shape[0] / 2)]
            center_radius = int(shape[1] / 2.8)
            circle_of_interest = [center_frame[0], center_frame[1], center_radius]

            # Check if center is inside the manually defined circle of interest:
            if major_axis > minimum_major_axis:

                if is_p_inside_circle(center, circle_of_interest):
                    if is_ellipse_new(markers, [center[0], center[1], major_axis]):
                        markers.append([center[0], center[1], major_axis])  # [x,y,radius]

                        cv2.line(drawing, (int(xtop), int(ytop)), (int(xbot), int(ybot)), (0, 0, 255), 1)
                        cv2.circle(drawing, (int(xtop), int(ytop)), 1, (0, 255, 255), 1)
                        cv2.circle(drawing, (int(xbot), int(ybot)), 1, (0, 255, 0), 1)
                        cv2.circle(drawing, (int(xc), int(yc)), 1, (255, 0, 0), 1)
                        num_ellipses += 1
                        print(major_axis)

    cv2.imshow('Ellipses contours', drawing)
    print("Found", num_ellipses, "ellipses!")

    return drawing, markers, num_ellipses

# Check if ellipse was not already detected
def is_ellipse_new(markers, new_marker):
    c_x, c_y, major_axis = new_marker
    th = 0.1
    for m in markers:
        m_c_x, m_c_y, m_major_axis = m
        if abs(m_c_x-c_x)<th and abs(m_c_y-c_y)<th and abs(m_major_axis-major_axis)<th:
            return False
    return True




def fit_ellipse(img, blob_radius):
    minimum_major_axis = 5  # min. pixel length of ellipse's major axis

    # Convert image to gray and blur it
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))
    #cv2.imshow("src_gray", src_gray)

    canny_threshold = 30  # initial threshold

    canny_output = cv2.Canny(src_gray, canny_threshold, canny_threshold*2)
    #cv2.imshow("canny_output", canny_output)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # Find the rotated rectangles and ellipses for each contour
    minRect = [None] * len(contours)
    minEllipse = [None] * len(contours)

    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv2.fitEllipse(c)
            (xc, yc), (d1, d2), angle = minEllipse[i]
            #print("Found ellipse [xc, yc, d1, d2, angle]:\n", xc, yc, d1, d2, angle)

            # Draw contours + rotated rects + ellipses (draw vertical line; compute major radius)
            rmajor = max(d1, d2) / 2
            if angle > 90:
                angle = angle - 90
            else:
                angle = angle + 90
            xtop = xc + math.cos(math.radians(angle)) * rmajor
            ytop = yc + math.sin(math.radians(angle)) * rmajor
            xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
            ybot = yc + math.sin(math.radians(angle + 180)) * rmajor
            # Check if center is inside the manually defined circle of interest:
            if rmajor < blob_radius*3:
                return xc, yc, rmajor*2, xtop, ytop, xbot, ybot
    return None, None, None, None, None, None, None


def detect_markers_circles(frame):
    detector = set_parameters()  #get the detector with set params
    overlay = frame.copy()  #copy the frame

    frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)  # binarize the frame (maybe not needed)
    #cv2.imshow("bin", frame_gray) # to show binarized frame

    keypoints = detector.detect(frame_gray)  # found circles (markers)

    im_with_keypoints = cv2.drawKeypoints(frame.copy(), keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Manually define a circle of interest
    shape = frame_gray.shape
    center_frame = [int(shape[1]/2), int(shape[0]/2)]
    center_radius = int(shape[1]/2.8)
    circle_of_interest = [center_frame[0], center_frame[1], center_radius]

    # draw the cirlce of interest + draw horizontal and vertical central lines
    cv2.circle(overlay, (center_frame[0], center_frame[1]), center_radius, (0, 255, 0), 5)
    cv2.line(overlay, (50, center_frame[1]), (shape[1]-50, center_frame[1]), (0, 255, 0), 1)  #vertical
    cv2.line(overlay, (center_frame[0], 50), (center_frame[0], shape[0]-50), (0, 255, 0), 1)  #horizontal

    # markers = list of lists containing center x, y coordinates and radius
    markers = []
    num_ellipses = 0
    for k in keypoints:
        c_x, c_y = k.pt[0], k.pt[1]
        radius = k.size/2

        # Check if center is inside the manually defined circle of interest:
        if is_p_inside_circle([c_x, c_y], circle_of_interest):
            #cv2.circle(overlay, (int(c_x), int(c_y)), int(radius), (0, 0, 255), -1)
            #cv2.line(overlay, (int(c_x) - 10, int(c_y)), (int(c_x) + 10, int(c_y)), (0, 0, 0), 2)
            #cv2.line(overlay, (int(c_x), int(c_y) - 10), (int(c_x), int(c_y) + 10), (0, 0, 0), 2)

            '''
            px_th = 3
            frame_to_crop = frame.copy()
            cropped_blob = frame_to_crop[int(c_y-radius-px_th):int(c_y+radius+px_th), int(c_x-radius-px_th):int(c_x+radius+px_th)]
            cv2.imshow("cropped", cropped_blob)
            xc, yc, major_axis, xtop, ytop, xbot, ybot = fit_ellipse(cropped_blob, radius)

            if xc is None: #if for whatever reason no ellipse is fitted
                markers.append([c_x, c_y, radius])  # [x,y,radius]
            else:
                num_ellipses += 1
                new_xc = c_x - radius - px_th + xc
                new_yc = c_y - radius - px_th + yc
                new_xtop = c_x - radius - px_th + xtop
                new_ytop = c_y - radius - px_th + ytop
                new_xbot = c_x - radius - px_th + xbot
                new_ybot = c_y - radius - px_th + ybot

                markers.append([new_xc, new_yc, major_axis/2])
                # ATTENTION: I'm returning center of blobs!
                #markers.append([c_x, c_y, major_axis/2])

                # Draw ellipses
                cv2.line(overlay, (int(new_xtop), int(new_ytop)), (int(new_xbot), int(new_ybot)), (255, 255, 255), 3)
                cv2.circle(overlay, (int(new_xtop), int(new_ytop)), 1, (0, 255, 255), 1)
                cv2.circle(overlay, (int(new_xtop), int(new_ybot)), 1, (0, 255, 0), 1)
                cv2.circle(overlay, (int(new_xc), int(new_yc)), 1, (255, 0, 0), 1)
            '''

            markers.append([c_x, c_y, radius])

            opacity = 0.5
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    num_detected_markers = len(markers)  #number of markers
    print("Detected", num_detected_markers, "markers!")
    return im_with_keypoints, markers, num_detected_markers



def is_p_inside_circle(p, circle):
    x,y = p
    center_x, center_y, radius = circle
    if (x - center_x)**2 + (y - center_y)**2 < radius**2:
        return True
    else:
        return False

