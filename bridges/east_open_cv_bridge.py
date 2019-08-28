"""
This file contains source code from another GitHub project. The comments made there apply. The source code
was licensed under the MIT License. The license text and a detailed reference can be found in the license
subfolder at models/east_open_cv/license. Many thanks to the author of the code.

For reasons of clarity unneeded parts of the original code were not taken over. The original project can
be found on the https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV page.

For a better understanding the documentation has been supplemented in parts. Code completely or predominantly
taken from the source was marked with "External code".
"""

import time

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

import bridges_config as config


class EastOpenCvBridge:
    """A bridge class for connecting to a text detector
    """

    def __init__(self):
        """The constructor
        """
        self.load_model()

    def load_model(self):
        """Loads the underlying model together with its pre-trained weights.
        """
        self.model = cv2.dnn.readNet(config.EAST_OPENCV_MODEL_PATH)

    def scann(self, image):
        """External code
        Examines the passed image for text regions and returns them as a collection of boxes in the
        form of a NumPy array. The passed image must be a raster image.

        :param image:The image to be examined.
        :return:A NumPy array of predicted text areas.
        """

        # load the input image and grab the image dimensions
        self.orig = image.copy()
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height, should be multiple of 32
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        self.layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        self.model.setInput(blob)
        (scores, geometry) = self.model.forward(self.layerNames)
        end = time.time()

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []  # stores the bounding box coordiantes for text regions
        confidences = []  # stores the probability associated with each bounding box region in rects

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < 0.5:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        """
        Extension to the original code to return a usable format.
        """
        newboxes = []

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            box = []
            box.append([startX, startY])
            box.append([endX, startY])
            box.append([endX, endY])
            box.append([startX, endY])

            newboxes.append(box)

        return np.asarray(newboxes)
