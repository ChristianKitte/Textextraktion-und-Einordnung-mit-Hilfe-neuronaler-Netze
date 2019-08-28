"""
This file contains source code from another GitHub project. The comments made there apply. The source code
was licensed under the MIT License. The license text and a detailed reference can be found in the license
subfolder at models/crnn/license. Many thanks to the author of the code.

For reasons of clarity unneeded parts of the original code were not taken over. The original project can
be found on the https://github.com/kurapan/CRNN page.

For a better understanding the documentation has been supplemented in parts. Code completely or predominantly
taken from the source was marked with "External code".
"""

import argparse
import string

import keras.backend as K

import bridges_config as config
from crnn.models import CRNN_STN
from crnn.utils import *


class CrnnBridge:
    """A bridge class for connecting to a text detector
    """

    def __init__(self):
        """The constructor
        """
        self.load_model()

    def load_model(self):
        """Generates the model based on the transferred parameters and loads the pre-trained weights.
        """
        self.model = CRNN_STN(self.crnn_cfg())
        self.model.load_weights(config.CRNN_Model_Path)

    def scann(self, image):
        """External code
        Examines the passed image and returns the predicted text. The passed image must
        be a raster image.

        :param image:The image to be examined
        :return:The predicted text as string.
        """
        img = self.preprocess_image(image)

        y_pred = self.model.predict(img[np.newaxis, :, :, :])

        # The CTC loss is calculated via Keras by TensoFlow.
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
        ctc_out = K.get_value(ctc_decode)[:, :self.crnn_cfg().label_len]

        result_str = ''.join([self.crnn_cfg().characters[c] for c in ctc_out[0]])
        result_str = result_str.replace('-', '')

        return result_str

    def crnn_cfg(self):
        """External code
        Defines in the original project a number of parameters among others for the
        definition of the model

        :return:The instance of an input value array
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--data_path', type=str, default='')
        parser.add_argument('--gpus', type=int, nargs='*', default=[0])
        parser.add_argument('--characters', type=str, default='0123456789' + string.ascii_lowercase + '-')
        parser.add_argument('--label_len', type=int, default=16)
        parser.add_argument('--nb_channels', type=int, default=1)
        parser.add_argument('--width', type=int, default=200)
        parser.add_argument('--height', type=int, default=31)
        parser.add_argument('--model', type=str, default='CRNN_STN', choices=['CRNN_STN', 'CRNN'])
        parser.add_argument('--conv_filter_size', type=int, nargs=7, default=[64, 128, 256, 256, 512, 512, 512])
        parser.add_argument('--lstm_nb_units', type=int, nargs=2, default=[128, 128])
        parser.add_argument('--timesteps', type=int, default=50)
        parser.add_argument('--dropout_rate', type=float, default=0.25)

        return parser.parse_args()

    def preprocess_image(self, img):
        """External code
        Carries out some pre-processing

        :param img:The image to be edited
        :return:The edited image
        """

        # if channel 1 then as grayscale
        if img.shape[1] / img.shape[0] < 6.4:
            img = pad_image(img, (self.crnn_cfg().width, self.crnn_cfg().height), self.crnn_cfg().nb_channels)
        else:
            img = resize_image(img, (self.crnn_cfg().width, self.crnn_cfg().height))
        if self.crnn_cfg().nb_channels == 1:
            img = img.transpose([1, 0])
        else:
            img = img.transpose([1, 0, 2])

        img = np.flip(img, 1)
        img = img / 255.0
        if self.crnn_cfg().nb_channels == 1:
            img = img[:, :, np.newaxis]
        return img
