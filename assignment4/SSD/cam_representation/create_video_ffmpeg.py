import cv2
import numpy as np
import glob

frameSize = (1024, 128)

out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)

for filename in glob.glob('/work/gianlh/TDT4265_StarterCode/assignment4/SSD/cam_representation/_task_2_3_3/val/*.png'):
    img = cv2.imread(filename)
    out.write(img)

out.release()
