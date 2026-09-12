# code for taking the camera feed, identifying the board, and rectifying the images using OpenCV.

import cv2 as cv
import src.camera as camera
import numpy as np

def process_frame(frame, max_contours, stabilizer = None):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = camera.edge_detection_thresholding(gray, (15, 15), 150, 250)
    contours = camera.approximate_contours(camera.find_contours(canny), max_contours)

    if stabilizer is not None:
        confirmed = stabilizer.update(contours)
    else:
        confirmed = []

    is_stable = [stabilizer is not None and stabilizer.matches_any(c, confirmed) for c in contours]
    stable = [c for c, s in zip(contours, is_stable) if s]
    candidates = [c for c, s in zip(contours, is_stable) if not s]

    return stable, candidates