"""A general configuration file for the bridges used. The naming convention is freely selectable. Each
bridge can store and behave entries here.
"""

GPU_LIST = '0'
"""Specification of the available GPUs
"""
CRNN_Model_Path = 'bridges/models/crnn/pretrained/model.hdf5'
"""Path to CRNN model (model and weights)
"""
EAST_MODEL_PATH = 'bridges/models/east/pretrained/model.h5'
"""Path to EAST model (model)
"""
EAST_JSON_PATH = 'bridges/models/east/pretrained/model.json'
"""Path to EAST model (weights)
"""
EAST_OPENCV_MODEL_PATH = 'bridges/models/east_open_cv/pretrained/frozen_east_text_detection.pb'
"""Path to the weights of the EAST model from Open CV
"""
