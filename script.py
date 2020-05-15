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
W, H = 224, 224
saved_path = 'URFD_images1/'

if not os.path.exists(saved_path):
	os.makedirs(saved_path)

while True:
    frame = vs.read()
    frame = cv2.resize(frame, (W,H))
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d:%m:%Y;%H:%M:%S.%f") 
    cv2.imwrite(saved_path + str(timestampStr) + '.jpg',cv2.resize(frame, (W,H)),[int(cv2.IMWRITE_JPEG_QUALITY), 95])
