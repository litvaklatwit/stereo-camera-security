#!/home/user/.virtualenvs/facecourse-py3/bin/python3
from imutils.video import VideoStream
import imutils
import cv2
import time

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

time.sleep(2.0)

while True:
    frameL = vsLeft.read()
    frameR = vsRight.read()

    if frameL is None or frameR is None:
        break

    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

vsLeft.stop()
vsRight.stop()
cv2.destroyAllWindows()