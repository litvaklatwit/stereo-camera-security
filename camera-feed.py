
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

import numpy as np

import pcl

#vsLeft = VideoStream(src=0, resolution=(1920, 1080), framerate=30).start()
#vsRight = VideoStream(src=6, resolution=(1920, 1080), framerate=30).start()
vsLeft = VideoStream(src="/dev/video2")
vsRight = VideoStream(src="/dev/video6")

CAPTURE_WIDTH = 640 # TODO: change and recalibrate to match
CAPTURE_HEIGHT = 480

vsLeft.stream.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
vsLeft.stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
vsLeft.stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

vsRight.stream.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
vsRight.stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
vsRight.stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

vsLeft.start()
vsRight.start()

Q = np.load("output/disp_to_depth_mat.npy")

undist_left = np.load("output/undistortion_map_left.npy")
undist_right = np.load("output/undistortion_map_right.npy")

rectif_left = np.load("output/rectification_map_left.npy")
rectif_right = np.load("output/rectification_map_right.npy")

trans_vec = np.load("output/trans_vec.npy")
rot_mat = np.load("output/rot_mat.npy")

rot_vec = cv2.Rodrigues(rot_mat)[0]
rot_vec = (rot_vec[0][0], rot_vec[1][0], rot_vec[2][0])

cam_mat = np.load("output/cam_mats_left.npy")
dist_coefs_left = np.load("output/dist_coefs_left.npy")

time.sleep(2.0)

st_1 = {
    "preFilterCap":1,
    "blockSize":5,
    "minDisparity":-28,
    "numDisparities":112,
    "P1":0,
    "P2":0,
    "uniquenessRatio":0,
    "speckleWindowSize":0,
    "disp12MaxDiff":100
}

st_2 = {
    "preFilterCap":8,
    "blockSize":9,
    "minDisparity":-27,
    "numDisparities":128,
    "P1":0,
    "P2":10000,
    "uniquenessRatio":9,
    "speckleWindowSize":0,
    "disp12MaxDiff":0
}

st_3 = {
    "preFilterCap":40,
    "blockSize":17,
    "minDisparity":30,
    "numDisparities":64,
    "P1":0,
    "P2":10000,
    "uniquenessRatio":24,
    "speckleWindowSize":0,
    "disp12MaxDiff":0,
}

st_4 = {
    "preFilterCap":42,
    "blockSize":7,
    "minDisparity":-93,
    "numDisparities":176,
    "P1":289,
    "P2":10000,
    "uniquenessRatio":15,
    "speckleWindowSize":28,
    "speckleRange":3,
    "disp12MaxDiff":-1
}

stereo = cv2.StereoSGBM_create(**st_4)

firstFrame = None

vis = pcl.pcl_visualization.PCLVisualizering()
vis.AddCoordinateSystem(1.0)
vis.InitCameraParameters()

def reject_outliers(data, m = 2):
    #return data[np.abs(data[:, 2] - np.mean(data[:, 2])) < m * np.std(data[:, 2])]
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

while not vis.WasStopped():
    #if not vsLeft.stream.stream.grab() and not vsRight.stream.stream.grab():
    #    break

    #_, frameL = vsLeft.stream.stream.retrieve()
    #_, frameR = vsRight.stream.stream.retrieve()

    frameL = vsLeft.read()
    frameR = vsRight.read()

    if frameL is None or frameR is None:
        break

    frameL = cv2.remap(frameL, undist_left, rectif_left, cv2.INTER_NEAREST)
    frameR = cv2.remap(frameR, undist_right, rectif_right, cv2.INTER_NEAREST)

    #frameL = imutils.resize(frameL, width=500)
    #frameR = imutils.resize(frameR, width=500)

    #frameL = cv2.pyrDown(frameL)
    #frameR = cv2.pyrDown(frameR)

    gray = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray

    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #frameRGray = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    dispRaw = stereo.compute(frameL, frameR).astype(np.float32) / 16.0

    disp = np.empty(dispRaw.shape, dtype=np.uint8)
    cv2.normalize(dispRaw * 16.0, disp, 0.1, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    h, w = frameL.shape[:2]
    f = 0.8*w                          # guess for focal length
    ##f = 0.1 * w
    #Q = np.float32([[1, 0, 0, -0.5*w],
    #                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
    #                [0, 0, 0,     -f], # so that y-axis looks up
    #                [0, 0, 1,      0]])

    points = cv2.reprojectImageTo3D(dispRaw, Q)
    #zDepth = points[:, :, 2]
    #zDepth = -zDepth

    mask_disp = dispRaw > dispRaw.min()
    mask_inf = ~(np.isinf(points[:,:,0]) | np.isinf(points[:,:,1]) | np.isinf(points[:,:,2]))
    mask_nan = ~(np.isnan(points[:,:,0]) | np.isnan(points[:,:,1]) | np.isnan(points[:,:,2]))

    mask = mask_disp & mask_inf & mask_nan

    #pm = points.ravel().reshape((640 * 480, 3))
    pm = points[mask]
    pm = pm[(pm[:, 2] < 0) & (pm[:, 2] > -20)]
    #pc = pcl.PointCloud(pm.shape[0])
    #np.asarray(pc)[:, :3] = pm 
    pc = pcl.PointCloud(pm)

    vis.RemoveAllPointClouds(0)
    vis.AddPointCloud(pc)
    vis.SpinOnce(force_redraw=True)

    #depthColors = cv2.projectPoints(points, rot_vec, trans_vec, cam_mat, dist_coefs_left)[1]
    #depthColors = depthColors.resize(frameL.shape)

    #depthColors = np.empty(points.shape[:2], dtype=np.uint8)
    #cv2.normalize(zDepth, depthColors, 0.1, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #depthColors = cv2.cvtColor(depthColors, cv2.COLOR_GRAY2BGR)

    #depthColors = cv2.cvtColor(points[:, :, 2] / 1024.0 + 0.25, cv2.COLOR_GRAY2BGR)
    #pointColors = cv2.cvtColor(points, cv2.COLOR_BGR2RGB)
    #colors = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
    #mask = dispRaw > dispRaw.min()
    #out_points = points[mask]
    #out_colors = colors[mask]

    frameL = frameL.astype(np.float64) / 255.0

    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:#args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frameL, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #rect = frameL[x:x + w - 1:1, y:y + h - 1:1, :]
        #rect = (points[y:y + h, x:x + w, :]).copy()#.astype(np.uint8)

        #if rect.shape[0] <= 0 or rect.shape[1] <= 0:
        #    continue

        #dist = rect[rect > rect.min()].mean()
        #mask = rect > rect.min()

        #if len(rect[mask]) > 1 and rect[mask].mean() > 0.5:
        #    frameL[y: y + h, x:x + w, :] = rect
        
        #if dist > 0.5:
        #    frameL[y:y + h, x:x + w, :] = rect

        #cv2.imshow("Rect " + str(i), rect)

    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)
    cv2.imshow("Disparity", disp)
    #cv2.imshow("Depth", depthColors)

    #cv2.imshow("Delta", frameDelta)
    #cv2.imshow("Gray", gray)
    #cv2.imshow("Thresh", thresh)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

vsLeft.stop()
vsRight.stop()
cv2.destroyAllWindows()