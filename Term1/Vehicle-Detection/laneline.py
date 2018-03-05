import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error

x_cor = 9  # Number of corners to find
y_cor = 6
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((y_cor * x_cor, 3), np.float32)
objp[:, :2] = np.mgrid[0:x_cor, 0:y_cor].T.reshape(-1, 2)


def camera_cal():
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('camera_cal/calibration*.jpg')  # Make a list of paths to calibration images
    # Step through the list and search for chessboard corners
    corners_not_found = []  # Calibration images in which opencv failed to find corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conver to grayscale
        ret, corners = cv2.findChessboardCorners(gray, (x_cor, y_cor), None)  # Find the chessboard corners
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            corners_not_found.append(fname)
    print
    'Corners were found on', str(len(imgpoints)), 'out of', str(len(images)), 'it is', str(
        len(imgpoints) * 100.0 / len(images)), '% of calibration images'
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


mtx, dist = camera_cal()

