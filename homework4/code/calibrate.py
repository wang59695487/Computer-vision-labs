import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import coloredlogs
import argparse
import matplotlib as mpl


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

# initailization for plotting and logging
# Setting up font for matplotlib
mpl.rc("font", family=["Josefin Sans", "Trebuchet MS", "Inconsolata"], weight="medium")

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

import logging
# Setting up logger for the project
log = logging.getLogger(__name__)

# Adding argument parser
parser = argparse.ArgumentParser(description="""Do a calibration using a bunch of checker board images in a folder provided by the user
User can choose to view the calibrated images or not
The camera instrinsics parameters and distortion coefficients are stored in a file whose name is specified by the user
""", formatter_class=CustomFormatter)
parser.add_argument("-c", "--columns", type=int, default=12)
parser.add_argument("-r", "--rows", type=int, default=12)



args = parser.parse_args()
# parse and process the parsed arguments
path = "./calibration"
fpaths = [os.path.join(path, fname) for fname in os.listdir(path)]  # all image paths
board_w = args.columns
board_h = args.rows
board_n = board_w*board_h
board_sz = (board_w, board_h)
output = "intrinsics.xml"

# initializating
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria for subPix corners finder
img_pts = []  # image points in calibrate camera
obj_pts = []  # world points in calibrate camera
img_shp = ()  # size/shape of the image to be used for calibrattion
objp = np.zeros((board_n, 3), "float32")  # all object/world points are the same set
objp[:, :2] = np.mgrid[0:board_h, 0:board_w].T.reshape(-1, 2)
imgs = []  # so we don't have to read all images in again
files = []  # valid filenames

# find all checker board corners and add corresponding 3D space locations
for fpath in fpaths:
    log.info(f"Begin processing {fpath}")
    img = cv2.imread(fpath)
    if img is None:
        log.warning(f"Cannot read image {fpath}, is this really a image?")
        break
    img_shp = img.shape[:2][::-1]  # OpenCV wants (width, height)
    found, corners = cv2.findChessboardCorners(
        img,  # the BGR image to be used to find checker board corners on
        board_sz,  # (board_w, board_h)
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        # do a normalization beforehand
        # CALIB_CB_ADAPTIVE_THRESH Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness).
        # use adaptive threashold to BW image instead of a fixed one
        # CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with equalizeHist before applying fixed or adaptive thresholding.
    )
    if not found:
        log.warning(f"Cannot find checker board from image {fpath}, is there really a checker board? Do your width/height match?")
        break

    log.info(f"Found {corners.shape[0]} checkerboard corners")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(
        gray,  # cornersSubPix only wants one-channel image
        corners,  # already found corners to be refined on
        (11, 11),  # winSize	Half of the side length of the search window. For example, if winSize=Size(5,5) , then a (5∗2+1)×(5∗2+1)=11×11 search window is used.
        (-1, -1),  # zeroZone Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. It is used sometimes to avoid possible singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.
        criteria
    )
    log.info(f"Refined {corners.shape[0]} checkerboard corners")
    imgs.append(img)
    files.append(fpath)
    img_pts.append(corners)
    obj_pts.append(objp)

# cv2.calibrateCamera only accepts float32 as numpy array
obj_pts = np.array(obj_pts).astype("float32")

log.info(f"Beginning calibration using images in: {path}, files: {files}")
# do the calibration
err, intr, dist, rota, tran = cv2.calibrateCamera(
    obj_pts,  # object points in 3D
    img_pts,  # corner points in 2D
    img_shp,  # shape of the image to be calibrated
    None,  # setting to None to let the function return them
    None,  # setting to None to let the function return them
    # flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_PRINCIPAL_POINT
    # CALIB_ZERO_TANGENT_DIST Tangential distortion coefficients (p1,p2) are set to zeros and stay zero
    # The principal point is not changed during the global optimization. It stays at the center
)

log.info(f"Got camera intrinsics:\n{intr} and distortion coefficients:\n{dist}")

log.info(f"Opening {output} for output")
# store camera intrinsics and distortion coefficients
fs = cv2.FileStorage(output, cv2.FILE_STORAGE_WRITE)
fs.write("image_width", img_shp[0])
fs.write("image_height", img_shp[1])
fs.write("camera_matrix", intr)
fs.write("distortion_coefficients", dist)
fs.release()
log.info(f"camera_matrix and distortion_coefficients stored to {output}")


# mapx, mapy = cv2.initUndistortRectifyMap(intr, dist, None, intr, img_shp, cv2.CV_16SC2)
for i in range(len(imgs)):
        img = imgs[i]
        corners = img_pts[i]
        file = files[i]
        log.info(f"Showing {file}")
        cv2.drawChessboardCorners(img, board_sz, corners, True)
        # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        dst = cv2.undistort(img, intr, dist)
        plt.subplot(121)
        # plt.figure("Original")
        plt.imshow(img[:, :, ::-1])
        plt.title("Original")
        plt.subplot(122)
        # plt.figure("Rectified")
        plt.imshow(dst[:, :, ::-1])
        plt.title("Rectified")
        plt.show()
