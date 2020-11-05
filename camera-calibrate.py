import numpy as np
import cv2
import time
from imutils.video import VideoStream

square_size = 1.0
pattern_size = (9, 6)

pattern_points = np.zeros(shape=(np.prod(pattern_size), 3), dtype=np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

vs = VideoStream(src="/dev/video0")

# TODO: set vs starting params, not important right now

vs.start()

time.sleep(2.0)

min_rms = 1000

while True:
    frame = vs.read()

    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = frame.shape[:2]
    obj_points = []
    img_points = []

    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

        newMtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newMtx)

        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        if dst.shape[0] > 0 and dst.shape[1] > 0 and rms < min_rms:
            min_rms = rms
            print(min_rms)
            cv2.imshow("Undistorted", dst)

    cv2.drawChessboardCorners(frame, pattern_size, corners, found)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break