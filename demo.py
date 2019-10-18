""" Function of demo.py
This script allows you to easily test the functionality of the solution using
images stored in the income directory.

The image stored there is transferred to an instance of the scanner, processed
and then returned as a new image and stored in the outcome directory.

The application of the auto function of the scanner class will also be demonstrated.
"""
import cv2

import constant as const
from annotation_constants.eval_annotation_constants import EVAL_ANNOTATION_CONTANTS
from annotation_constants.neg_annotation_constants import NEG_ANNOTATION_CONTANTS
from annotation_constants.pos_annotation_constants import POS_ANNOTATION_CONTANTS
from scanner import Scanner


def auto(input, output):
    """Examines the image defined as input (path + file name) and generates an output image at
    the location defined as output (path + file name).

    :param input:Path to the input image
    :param output:Path to the output image
    """
    try:
        scanner = Scanner()
        scanner.auto_scann(input, output,
                           pos_annotation_constants=POS_ANNOTATION_CONTANTS,
                           neg_annotation_constants=NEG_ANNOTATION_CONTANTS,
                           eval_annotation_constants=EVAL_ANNOTATION_CONTANTS)
    except:
        print('Error in method {0} in module {1}'.format('auto', 'demo.py'))


if __name__ == '__main__':
    """Is executed when the file is executed directly. The function performs a text detection and 
    recognition with the image stored in the variable in_file and stores the result under the name 
    defined in the variable out_file.

    Version 1 takes over the loading and saving. Only the paths are to be specified.

    In Version2, the loading and saving processes must be performed manually. Detail screens are 
    output during processing
    """

    # Controls the program run
    version = 2
    # Input file (from the income directory)
    in_file = 'test.jpg'
    # Output file (from the outcome directory)
    out_file = '001.jpg'

    try:
        if version == 1:
            auto(const.INPUT_DIR + '/' + in_file, const.OUTPUT_DIR + '/' + out_file)
            exit(0)
        elif version == 2:
            scanner = Scanner()

            img_in = cv2.imread(const.INPUT_DIR + '/' + in_file)[:, :, ::-1]

            if img_in is not None:
                img_out = scanner.scann(img=img_in, print_detail=True,
                                        pos_annotation_constants=POS_ANNOTATION_CONTANTS,
                                        neg_annotation_constants=NEG_ANNOTATION_CONTANTS,
                                        eval_annotation_constants=EVAL_ANNOTATION_CONTANTS)
                cv2.imwrite(const.OUTPUT_DIR + '/' + out_file, img_out)

                print('Finished')
            else:
                print('Image not readable')
    except:
        print('Error in method {0} in module {1}'.format('main', 'demo.py'))
