"""
This file contains source code from another GitHub project. The comments made there apply. The source code
was licensed under the GNU General Public License v3.0. The license text and a detailed reference can be
found in the license subfolder at models/east/license. Many thanks to the author of the code.

For reasons of clarity unneeded parts of the original code were not taken over. The original project can
be found on the https://github.com/kurapan/EAST page.

For a better understanding the documentation has been supplemented in parts. Code completely or predominantly
taken from the source was marked with "External code".
"""

import sys

import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

import bridges_config as config

sys.path.append('/bridges/models')
import east.lanms as east_lanms
import east.model as east_model
from data_processor import restore_rectangle


class EastBridge:
    """A bridge class for connecting to a text detector
    """

    def __init__(self):
        """The constructor
        """
        self.load_model()

    def load_model(self):
        """Loads the underlying model and the pre-trained weights.
        """
        json_file = open(config.EAST_JSON_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json,
                                     custom_objects={'tf': tf, 'RESIZE_FACTOR': east_model.RESIZE_FACTOR})
        self.model.load_weights(config.EAST_MODEL_PATH)

    def scann(self, image):
        """External code
        Examines the passed image for text regions and returns them as a collection of boxes in the
        form of a NumPy array. The passed image must be a raster image.

        :param image:The image to be examined
        :return:A NumPy array of predicted text areas.
        """
        img_resized, (ratio_h, ratio_w) = self.resize_image(image)
        img_resized = (img_resized / 127.5) - 1

        score_map, geo_map = self.model.predict(img_resized[np.newaxis, :, :, :])

        boxes = self.detect(score_map=score_map, geo_map=geo_map)

        new_boxes = []

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

            for box in boxes:
                # to avoid submitting errors
                box = self.sort_poly(box.astype(np.int32))

                """
                Extension to the original code to avoid errors.
                """
                # if condition is met, the distance is too small, then next
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue

                new_boxes.append(box)

        return new_boxes

    def resize_image(self, im, max_side_len=2400):
        """External code
        Resize image to a size multiple of 32 which is required by the network

        :param im:The resized image
        :param max_side_len:Limit of max image size to avoid out of memory in gpu
        :return:The resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
        im = cv2.resize(im, (int(resize_w), int(resize_h)))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)

    def detect(self, score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
        """External code
        Restore text boxes from score map and geo map

        :param score_map:List of probabilities
        :param geo_map:List of localities
        :param score_map_thresh:Threshhold for score map
        :param box_thresh:Threshhold for boxes
        :param nms_thres:Threshold for nms
        :return:The recognized regions as boxes
        """
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]

        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)

        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]

        # restore
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4,
                                              geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2

        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

        # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        boxes = east_lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

        if boxes.shape[0] == 0:
            return None

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes

    def sort_poly(self, p):
        """External code
        Sorts the polygon

        :param p:The polygon to be sorted
        :return:The sorted polygon
        """
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]
