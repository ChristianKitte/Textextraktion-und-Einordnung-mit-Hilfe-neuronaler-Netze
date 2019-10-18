import json


class Detector:
    """Represents an abstract detector. Provides functionality to detect scene text in images (as np array).
    The module and the class name of a bridge are transferred to a real model. The bridge class must have a
    parameterless constructor and a method named scan. The scan method passes the image to be analyzed as the
    only parameter. It returns a list of boxes (each box defined with four points).
    """

    def __init__(self, module_name, class_name):
        """The constructor.

        :param module_name:A module name that refers to a bridge module.
        :param class_name:A class name of a bridge from the specified bridge module.
        """
        try:
            module = __import__(module_name)
            my_class = getattr(module, class_name)

            self.instance = my_class()
        except:
            print('Error in method {0} in module {1}'.format('init', 'detector.py'))

    @staticmethod
    def instance(json_path):
        """Returns an instance of the class Detector.

        :param json_path:Path to a JSON file that defines the bridges.
        :return:A new instance of the class.
        """
        try:
            with open(json_path, mode='r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)
                detector_name = json_data['detector_module']
                detector_class = json_data['detector_class']

            return Detector(detector_name, detector_class)
        except:
            print('Error in method {0} in module {1}'.format('instance', 'detector.py'))
            return None

    def scann(self, image):
        """Examines the passed image by passing the image to the current bridge of the class.

        :param image:The image (as np array) to be examined
        :return:A list of boxes (each box defined with four points).
        """
        try:
            return self.instance.scann(image)
        except:
            print('Error in method {0} in module {1}'.format('scann', 'detector.py'))
            return None
