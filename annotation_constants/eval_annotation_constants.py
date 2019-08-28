import cv2


class EVAL_ANNOTATION_CONTANTS():
    """A class for defining the text output of the annotation for texts recognized in images, which
    identify ingredients via their static methods.
    """

    @staticmethod
    def FONT():
        """Defines the font.
        (Example: cv2.FONT_HERSHEY_SIMPLEX)

        :return:The font
        """
        return cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def SCALE():
        """Defines the scaling in relation to the base size of the font.
        (Example: 0.50)

        :return:The scale
        """
        return 0.50

    @staticmethod
    def THICKNESS():
        """Defines the thickness of the font.
        (Example: 1)

        :return:The thickness
        """
        return 1

    @staticmethod
    def COLOR():
        """Defines the color of the font as BGR.
        (Example: (0, 0, 255))

        :return:The color as BGR
        """
        return (0, 0, 255)  # BGR!

    @staticmethod
    def BORDER_COLOR():
        """Defines the color of the border as BGR.
        (Example: (0, 0, 255))

        :return:The color as BGR
        """
        return (0, 0, 255)  # BGR!

    @staticmethod
    def BORDER_THICKNESS():
        """Defines the thickness of the border.
        (Example: 1)

        :return:The thickness
        """

        return 2
