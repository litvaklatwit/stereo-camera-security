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

firstFrame = None

while True:
    frameL = vsLeft.read()
    frameR = vsRight.read()

    if frameL is None or frameR is None:
        break

    gray = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (21, 21), 0)
    gray = cv2.GaussianBlur(gray, (31, 31), 0)

    if firstFrame is None:
        firstFrame = gray

    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    outFrame = frameL.copy()

    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 10000:#500:#args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(outFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.drawContours(outFrame, cnts, -1, (255, 255, 0))

    #if firstFrame is not None:
    #    firstFrame = gray

    cv2.imshow("Left", outFrame)
    cv2.imshow("Right", frameR)
    cv2.imshow("Delta", frameDelta)
    cv2.imshow("Thresh", thresh)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

vsLeft.stop()
vsRight.stop()
cv2.destroyAllWindows()