#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2

vLeft = cv2.VideoCapture("./left.mpeg")
vRight = cv2.VideoCapture("./right.mpeg")

if __name__ == '__main__':
    print('loading images...')
    imgL = cv2.pyrDown(vLeft.read()[1])  # downscale images for faster processing
    imgR = cv2.pyrDown(vRight.read()[1])

    #imgL = vLeft.read()[1]
    #imgR = vRight.read()[1]

    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # disparity range is tuned for 'aloe' image pair
    window_size = 5 # 3 # relates to the size of blobs
    min_disp = 16 # 16 # seems to relate to the number of false positives
    num_disp = 112-min_disp

    # also relates to the size of blobs
    spec_size = 150 # 100, 0 # lower = more false positive blobs on the test image
    spec_range = 32 # 32, 20 # higher range = better capture of the cube for the test image

    unique_ratio = 5 # 10, 5
    disp12_max_diff = 64 # 1, 64

    #stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    #    numDisparities = num_disp,
    #    blockSize = window_size,
    #    uniquenessRatio = 10,
    #    speckleWindowSize = 100,
    #    speckleRange = 32,
    #    disp12MaxDiff = 1,
    #    P1 = 8*3*window_size**2,
    #    P2 = 32*3*window_size**2,
    #)

    stereo = cv2.StereoBM_create()
    stereo.setBlockSize(window_size) # block size seems to help with noise
    stereo.setNumDisparities(num_disp)
    stereo.setPreFilterSize(5)
    stereo.setPreFilterCap(1)
    stereo.setMinDisparity(min_disp)
    stereo.setTextureThreshold(5) # 5
    stereo.setUniquenessRatio(unique_ratio)
    stereo.setSpeckleWindowSize(spec_size)
    stereo.setSpeckleRange(spec_range)
    stereo.setDisp12MaxDiff(disp12_max_diff)

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    h, w = imgL.shape[:2]
    #f = 0.8*w                          # guess for focal length
    f = 0.1 * w
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()
