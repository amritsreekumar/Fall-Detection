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
saved_path = 'URFD_images/'
output_path = 'URFD_opticalflow/'
if not os.path.exists(saved_path):
	os.makedirs(saved_path)

if not os.path.exists(output_path):
	os.makedirs(output_path)



while True:
    frame = vs.read()
    frame = cv2.resize(frame, (W,H))
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d:%m:%Y;%H:%M:%S.%f") 
    cv2.imwrite(saved_path + str(timestampStr) + '.jpg',cv2.resize(frame, (W,H)),[int(cv2.IMWRITE_JPEG_QUALITY), 95])
    path = saved_path + '/'
	#vid_file = data_folder + activity_folder  + '/Videos/' + video_folder.replace(' ','')
    flow = output_path + '/'
    os.system('dense_flow2/build/extract_cpu -f={} -x={} -y={} -i=tmp/image -b=20 -t=1 -d=0 -s=1 -o=dir'.format(path, flow + '/flow_x', flow + '/flow_y'))
