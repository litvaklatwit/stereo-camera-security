#!/home/user/.virtualenvs/facecourse-py3/bin/python3
# NOTE: THE CHECKERBOARD SIZE IS 2.3CM
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time

import cv2

import numpy as np

#import pcl

import math

regress_A = 0.37154412
regress_B = 0.098914

def box_overlap(a, b):
    c = np.vstack([a, b])
    return np.hstack([np.max(c[:, :2], axis=0), np.min(c[:, 2:], axis=0)])

def depth_of(z):
    return np.log(z / regress_B) / regress_A

class DepthRegion:
    def __init__(self, side_length):
        self.side_length = side_length
        self.show = False
        self.selected = False
        self.point = (0, 0)

    def on_mouse(self, event, x, y):
        if event != cv2.EVENT_MOUSEMOVE and event != cv2.EVENT_LBUTTONDOWN:
            return

        self.show = True

        if event == cv2.EVENT_MOUSEMOVE:
            if not self.selected:
                self.point = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.selected = x >= self.point[0] - self.side_length and x <= self.point[0] + self.side_length and y >= self.point[1] - self.side_length and y <= self.point[1] + self.side_length
            self.point = (x, y)

    def show_elems(self, depth, elem_space=60):
        if not self.show:
            return

        n = 2 * self.side_length + 1
        im = np.ones((elem_space * n, elem_space * n, 3), dtype=np.uint8) * 255

        #mouse_img_cor = np.array([self.point[0], self.point[1], 1.0])
        #z = depth[self.point[1], self.point[0]]

        for i in range(-self.side_length, n + 1):
            x = self.point[0] + i

            if x < 0 or x >= depth.shape[1]:
                continue

            for j in range(-self.side_length, n + 1):
                y = self.point[1] + j

                if y < 0 or y >= depth.shape[0]:
                    continue

                color = (0, 0, 255) if i == 0 and j == 0 else (0, 0, 0)

                s = "{0:.2f}".format(depth_of(-depth[y, x, 2]) * 12 - 5)
                sz = cv2.getTextSize(s, cv2.FONT_HERSHEY_PLAIN, 1, 1)

                cv2.putText(im, s,
                        (int((i + self.side_length) * elem_space + (elem_space - sz[0][0]) / 2),
                        int((j + self.side_length) * elem_space + (elem_space + sz[0][1]) / 2)),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        cv2.imshow("Region", im)

    def draw_rect(self, img):
        if not self.show:
            return

        n = self.side_length if self.side_length > 1 else 1
        n = n + 1

        cv2.rectangle(img, (self.point[0] - n, self.point[1] - n),
                (self.point[0] + n, self.point[1] + n),
                (0, 255, 0) if self.selected else (0, 0, 255), 1)


dr = DepthRegion(3)

vsLeft = VideoStream(src="/dev/video2")
vsRight = VideoStream(src="/dev/video6")

CAPTURE_WIDTH = 640 # TODO: change and recalibrate to match
CAPTURE_HEIGHT = 480

min_area = 10000

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

regions = [np.array([20, 20, 200, 200])]

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

stereo = cv2.StereoSGBM_create(**st_2, mode=1)
stereoR = cv2.ximgproc.createRightMatcher(stereo)

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(80000)
wls_filter.setSigmaColor(1.8)

firstFrame = None

#vis = pcl.pcl_visualization.PCLVisualizering()
#vis.AddCoordinateSystem(1.0)
#vis.InitCameraParameters()

def reject_outliers(data, m = 2):
    #return data[np.abs(data[:, 2] - np.mean(data[:, 2])) < m * np.std(data[:, 2])]
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def onMousePressed(event, x, y, flags, userdata):
    dr.on_mouse(event, x, y)

kernel = np.ones((3, 3), dtype=np.uint8)

#while not vis.WasStopped():
while True:
    frameL = vsLeft.read()
    frameR = vsRight.read()

    if frameL is None or frameR is None:
        break

    frameL = cv2.remap(frameL, undist_left, rectif_left, cv2.INTER_NEAREST)
    frameR = cv2.remap(frameR, undist_right, rectif_right, cv2.INTER_NEAREST)

    gray = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (31, 31), 0)

    if firstFrame is None:
        firstFrame = gray

    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    dispL = stereo.compute(frameL, frameR)
    dispR = stereoR.compute(frameR, frameL)

    dispL = np.int16(dispL)
    dispR = np.int16(dispR)

    filtered_img = wls_filter.filter(dispL, frameL, None, dispR)#.astype(np.uint8)
    #filtered_img = cv2.normalize(src=filtered_img, dst=filtered_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    #filtered_img = np.uint8(filtered_img)

    filt_color = cv2.cvtColor(filtered_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)#cv2.applyColorMap(filtered_img, cv2.COLORMAP_OCEAN)

    points = cv2.reprojectImageTo3D(filtered_img, Q)
    
    h, w = frameL.shape[:2]

    for i, c in enumerate(cnts):
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frameL, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(frameL, cnts, i, (255, 255, 0))

        rect = points[y:y + h, x:x + w, :]
        disp_rect = filtered_img[y:y + h, x:x + w]

        bounding_rect = np.array([x, y, x + w, y + h])

        for region in regions:
            olap = box_overlap(bounding_rect, region)
            if olap[2] - olap[0] > 0 and olap[3] - olap[1] > 0:
                x_center = (olap[2] + olap[0]) // 2
                y_center = (olap[3] + olap[1]) // 2

                #olap_pts = points[thresh]
                #olap_pts = points[olap, 2].ravel()
                #olap_pts = olap_pts[~np.isnan(olap_pts)]
                #olap_pts = olap_pts[olap_pts > filtered_img.min()]
                #olap_pts = reject_outliers(olap_pts, m=3)

                #depth = olap_pts.mean()
                
                depth = points[x_center, y_center, 2]
                depth = depth_of(depth)

                cv2.rectangle(frameL, tuple(olap[:2]), tuple(olap[2:]), (255, 0, 0), 2)
                cv2.putText(frameL, str(depth), (x_center, y_center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #print(points[(olap[2] + olap[0]) // 2, (olap[3] - olap[1]) // 2])

    #print(points[int(h / 2), int(w / 2), 2])
    
    #vis.RemoveAllPointClouds(0)

    #pc = pcl.PointCloud(points[filtered_img > filtered_img.min()])
    #vis.AddPointCloud(pc, str.encode("Hello"))

    #vis.SpinOnce(force_redraw=True)

    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)

    dr.show_elems(points, 90)
    dr.draw_rect(filt_color)

    cv2.imshow("filt_color", filt_color)
    cv2.setMouseCallback("filt_color", onMousePressed)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

vsLeft.stop()
vsRight.stop()
cv2.destroyAllWindows()