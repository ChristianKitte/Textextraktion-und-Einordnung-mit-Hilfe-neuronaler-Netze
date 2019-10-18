import cv2
import numpy as np

import constant as const
from bounding_box_image_handler import BoundingBoxImageHandler as box_handler
from detector import Detector
from ingrediens import Ingredients
from recognizer import Recognizer


class Scanner:
    """The central class scanner controls the entire application and generates an OCR pipeline.
    The neural networks used are defined via a JSON file. The storage location of the JSON file is
    stored as a constant in BRIDGES_JSON.
    """

    def __init__(self, refresh_db=False, usePatch=False):
        """The constructor.

        :param refresh_db:If True, the database is updated using the stored Excel file.
        :param usePatch:If true, umlauts are treated as a, o and u
        """
        try:
            self.detector = Detector.instance(const.BRIDGES_JSON)
            self.recognizer = Recognizer.instance(const.BRIDGES_JSON)

            # Create database of ingredients, transfer excel data beforehand
            if refresh_db == True:
                Ingredients.convert(const.DATABASE_EXCEL, const.DATABASE_JSON)

            self.db = Ingredients.instance(const.DATABASE_JSON, usePatch=usePatch)
        except:
            print('Error in method {0} in module {1}'.format('init', 'scanner.py'))

    def auto_scann(self, input_file, output_file, pos_annotation_constants=None, neg_annotation_constants=None,
                   eval_annotation_constants=None):
        """Automatically performs all text recognition and ingredient matching steps.

        :param input_file:The input image (path).
        :param aoutput_file:The output image (path).
        :param pos_annotation_constants:Defining the text output of a positive annotation. Default = None.
        :param neg_annotation_constants:Defining the text output of a negative annotation. Default = None.
        :param eval_annotation_constants:Defining the text output for evaluation purposes. Default = None.
        """
        try:
            img_in = cv2.imread(input_file)[:, :, ::-1]

            if img_in is not None:
                img_out = self.scann(img=img_in,
                                     pos_annotation_constants=pos_annotation_constants,
                                     neg_annotation_constants=neg_annotation_constants,
                                     eval_annotation_constants=eval_annotation_constants)
                cv2.imwrite(output_file, img_out)
        except:
            print('Error in method {0} in module {1}'.format('auto_scann', 'scanner.py'))

    def scann(self, img, evaluation_mode=False, print_detail=False, print_format='jpg', small_annotation=True,
              pos_annotation_constants=None, neg_annotation_constants=None, eval_annotation_constants=None):
        """Performs text recognition and matching with ingredients. As a result, the image extended by bounding
        boxes is returned.

        :param img:The image to be examined (opened with Open CV as a numpy array).
        :param evaluation_mode:If set, the frame will be thicker and every recognized Word will be displayed in
        the image.
        :param print_detail:If true, all detail screens are output. The file name is the recognized
        text. Default = False.
        :param print_format:The format of the partial output as ending without dot. Default = jpg.
        :param pos_annotation_constants:Defining the text output of a positive annotation. Default = None.
        :param neg_annotation_constants:Defining the text output of a negative annotation. Default = None.
        :param eval_annotation_constants:Defining the text output for evaluation purposes. Default = None.
        :return:The image extended by bounding boxes.
        """
        try:
            boxes = self.detector.scann(img)

            #
            # If you draw step-by-step on only one image and additionally generate details for prediction from
            # this image, it can happen that the generated detail images and thus the results are falsified.
            #
            # Therefore, the detail images are generated using a copy.
            #
            outimg = np.copy(img)
            #

            pos_annotation_constants = pos_annotation_constants
            neg_annotation_constants = neg_annotation_constants
            eval_annotation_constants = eval_annotation_constants

            if boxes is not None:
                for box in boxes:
                    # Determine a drawing file for each box
                    detail_img = box_handler.get_subimage(outimg, box, greyscale=True, save=False)

                    # Predict a text for each drawing file.be careful about greyscale.
                    detail_txt = self.predict_text(detail_img, greyscale=False)

                    # Output single images, if desired
                    if print_detail:
                        cv2.imwrite(const.OUTPUT_DIR + '/' + detail_txt + '.' + print_format, detail_img)

                    # Visualize (enter bounding boxes)
                    if evaluation_mode == False:
                        # Test whether it is an ingredient
                        present, id, identification = self.db_contains(detail_txt)

                        if present == True:
                            if (small_annotation):
                                detail_name = self.db.get_enumber(id)
                            else:
                                detail_name = identification

                            box_handler.put_border(img, box,
                                                   box_handler.BGR_to_RGB(pos_annotation_constants.BORDER_COLOR()))
                            box_handler.put_text(img, box, detail_name, pos_annotation_constants)
                        else:
                            box_handler.put_border(img, box,
                                                   box_handler.BGR_to_RGB(neg_annotation_constants.BORDER_COLOR()))

                    if evaluation_mode == True:
                        detail_name = detail_txt

                        box_handler.put_border(img, box,
                                               box_handler.BGR_to_RGB(eval_annotation_constants.BORDER_COLOR()),
                                               eval_annotation_constants.BORDER_THICKNESS())
                        box_handler.put_text(img, box, detail_name, eval_annotation_constants)

            return img[:, :, ::-1]
        except:
            print('Error in method {0} in module {1}'.format('scann', 'scanner.py'))
            return None

    def predict_text(self, img, greyscale=True):
        """Uses the recognizer currently stored in the system to predict the text passed in the image.

        :param img:An image with contained text
        :param greyscale:If True, the passed image is converted to grayscale. Default is True.
        :return:The predicted text
        """
        try:
            if greyscale == True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            return self.recognizer.scann(img)
        except:
            print('Error in method {0} in module {1}'.format('predict_text', 'scanner.py'))
            return None

    def db_contains(self, searchstring):
        """Checks whether an ingredient exists using the transferred string. If it exists, the
        return is as follows:

            present=True
            id the id of the substance
            identification contains the textual description of the substance

        If the substance is not present:

            present=False
            id=-1
            identification=not available

        :param searchstring:The search string
        :return:present, id, identification
        """
        try:
            present, id = self.db.contains(searchstring)

            if present:
                identification = self.db.get_enumber(id) + ' - ' + self.db.get_name(id)[0]
                return present, id, identification
            else:
                return present, id, 'nicht vorhanden'
        except:
            print('Error in method {0} in module {1}'.format('db_contains', 'scanner.py'))
            return None
