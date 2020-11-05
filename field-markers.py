#!/home/user/.virtualenvs/facecourse-py3/bin/python3
from imutils.video import VideoStream
import imutils
import cv2
import time

import numpy as np

regions = [np.array([20, 20, 200, 200])]
test_rect = np.array([100, 100, 300, 300])

def box_overlap(a, b):
    c = np.vstack([a, b])
    return np.hstack([np.max(c[:, :2], axis=0), np.min(c[:, 2:], axis=0)])

vsLeft = VideoStream(src="/dev/video4")
vsRight = VideoStream(src="/dev/video0")

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

time.sleep(2.0)

while True:
    frameL = vsLeft.read()
    frameR = vsRight.read()

    if frameL is None or frameR is None:
        break

    cv2.rectangle(frameL, tuple(test_rect[:2]), tuple(test_rect[2:]), (0, 255, 0), 2)

    for region in regions:
        cv2.rectangle(frameL, tuple(region[:2]), tuple(region[2:]), (255, 255, 255), 2)
        sect = box_overlap(test_rect, region)
        cv2.rectangle(frameL, tuple(sect[:2]), tuple(sect[2:]), (255, 0, 0), 2)

    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

vsLeft.stop()
vsRight.stop()
cv2.destroyAllWindows()