import cv2
import os
import csv
import sys
import glob
import numpy as np
import imutils
import time
from datetime import datetime
from imutils.video import WebcamVideoStream
from imutils.video import FPS


vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
scale = 224
W, H = 224, 224
output_path = 'URFD_opticalflow/'

if not os.path.exists(output_path):
	os.makedirs(output_path)

first_frame = vs.read()
first_frame = cv2.resize(first_frame, (W,H))

# Convert to gray scale 
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)


# Create mask
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

while True:
    frame = vs.read()
    frame = cv2.resize(frame, (W,H))
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d:%m:%Y;%H:%M:%S.%f") 
    #cv2.imwrite(saved_path + str(timestampStr) + '.jpg',cv2.resize(frame, (W,H)),[int(cv2.IMWRITE_JPEG_QUALITY), 95])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Calculate dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    # Open a new window and displays the output frame
    dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
    cv2.imwrite(output_path + str(timestampStr) + '.jpg', dense_flow)
    #out.write(dense_flow)
    # Update previous frame
    prev_gray = gray
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

