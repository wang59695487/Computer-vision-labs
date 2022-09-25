import numpy as np
import cv2
import os
import glob as gb
import sys

if len(sys.argv) > 1:
    dirname = sys.argv[1]
else:
    dirname = "./resources/"

## load video
videoFile = gb.glob(dirname+"*.avi")
capture = cv2.VideoCapture(videoFile[0])
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = capture.get(cv2.CAP_PROP_FPS)
framecount = capture.get(cv2.CAP_PROP_FRAME_COUNT)

## load images
Images=[]
imgfile = gb.glob(dirname+"*.jpg")
for file in imgfile:
    img = cv2.imread(file)
    Images.append(img)

## resize images
for i in range(len(Images)):
    Images[i] = cv2.resize(Images[i], (int(width),int(height)),
                           interpolation=cv2.INTER_NEAREST)

## write videofile
videoW = cv2.VideoWriter(dirname+'output.avi',
                         cv2.VideoWriter_fourcc('M','J','P','G'), int(fps),
                        (int(width),int(height)),True)
fps1 = int(fps)

## write images
text = '%s %s' % ('3170100186', 'WANG JUN')
WAIT = fps1
for i in range(len(Images)*fps1*2):
    num = i//(fps1*2)
    if num<len(Images)-2:
        weight = (i-num*fps1*2) / WAIT
        img = cv2.putText(Images[num], text, (int(width/2-200),int(height-20)),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (246,178,107), 4)
        img1 = cv2.putText(Images[num+1], text, (int(width/2-200),int(height-20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (246,178,107), 4)
        res = cv2.addWeighted(img, 1-weight, img1, weight, 0)
    else:
        res = cv2.putText(Images[num-1], text, (int(width/2-200),int(height-20)),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (246,178,107), 4)
    videoW.write(res)


# write video
while(True):
    ret,video = capture.read()
    if ret:
        cv2.putText(video, text, (int(width/2-200),int(height-20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (246,178,107), 4)
        videoW.write(video)
    else:
        break
videoW.release()
