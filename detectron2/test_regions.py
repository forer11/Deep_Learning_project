import numpy as np
import cv2

# this is the model we'll be using for
# object detection
from tensorflow.keras.applications import Xception

# for preprocessing the input
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
# from imutils.object_detection import non_max_suppression

img = cv2.imread("./input.jpg")
(H, W) = img.shape[:2]

# instanciate the selective search
# segmentation algorithm of opencv
search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# set the base image as the input image
search.setBaseImage(img)

# since we'll use the fast method we set it as such
search.switchToSelectiveSearchFast()

# you can also use this for more accuracy:
# search.switchToSelectiveSearchQuality()
rects = search.process()  # process the image

roi = img.copy()
for (x, y, w, h) in rects:

    # Check if the width and height of
    # the ROI is atleast 10 percent
    # of the image dimensions and only then
    # show it
    if (w / float(W) < 0.1 or h / float(H) < 0.1):
        continue

    # Let's visualize all these ROIs
    cv2.rectangle(roi, (x, y), (x + w, y + h),
                  (0, 200, 0), 2)

roi = cv2.resize(roi, (640, 640))
final = cv2.hconcat([cv2.resize(img, (640, 640)), roi])
cv2.imshow('ROI', final)
cv2.waitKey(0)