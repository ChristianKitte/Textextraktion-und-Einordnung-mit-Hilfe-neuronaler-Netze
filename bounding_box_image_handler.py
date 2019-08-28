import os

import cv2
import numpy as np


class BoundingBoxImageHandler:
    """A class of static methods for creating, processing, and editing images with bounding boxes. As a rule, an
    image is regarded as a numpy array and processed as such. This allows efficient processing of raster graphics.
    In addition, an image is usually expected in this format.

    Permitted image formats for reading from or saving to a file depend on the data formats available in Open CV
    and are derived from the name via the specified extension:

        Windows bitmaps - *.bmp, *.dib (always supported)
        JPEG files - *.jpeg, *.jpg, *.jpe (see the Notes section within Open CV)
        JPEG 2000 files - *.jp2 (see the Notes section within Open CV)
        Portable Network Graphics - *.png (see the Notes section within Open CV)
        WebP - *.webp (see the Notes section within Open CV)
        Portable image format - *.pbm, *.pgm, *.ppm (always supported within Open CV)
        Sun rasters - *.sr, *.ras (always supported)
        TIFF files - *.tiff, *.tif (see the Notes section within Open CV)
    """

    @staticmethod
    def subimage_generator(image, bounding_boxes, start_at=1, zeros=3, output_dir='output', format='png'):
        """Enables the automated processing of multiple boxes based on the image being transferred. Uses the
        get_subimage method internally.

        The generated images are stored at the location stored in outpu_dir. The naming is done with numbers,
        starting from a starting number with leading zeros.

        :param image:The image from which a section is to be made.
        :param bounding_boxes:The boxes that define the areas. Each box creates a section. Could be a polygon.
        :param start_at:The starting number
        :param zeros:The number of leading zeros
        :param output_dir:The output directory
        :param format:The format to be used using the extension without a dot
        """
        img = cv2.imread(image)

        if img is not None:
            cv2.imwrite(output_dir + '/master.' + format, img)
            master_file = open(output_dir + '/master.txt', 'w')

            boxes = bounding_boxes
            counter = start_at

            if len(boxes) > 0:
                for box in boxes:
                    BoundingBoxImageHandler.get_subimage(img, box, save_as=output_dir + '/' + str(counter).zfill(
                        zeros) + '.' + format)
                    master_file.write(str(counter).zfill(3) + '.' + format + '\n')
                    counter = counter + 1

            master_file.close()

    @staticmethod
    def get_subimage(image, box, greyscale=False, save=True, save_as='new.jpg', show=False):
        """Extracts a section of the transferred image and returns it as an image (np array). The section is
        defined by the passed box. Optionally, sections can be saved, converted to grayscale or displayed in
        a preview.

        If the shape defined in the box is a polygon, the surrounding rectangle is used as the image section.

        :param image:The image from which a section is to be made.
        :param box:A box that defines the section. Could be a polygon.
        :param greyscale:True if the output is to be converted to grayscale, otherwise False.
        :param save:True if the output is to be saved, otherwise False.
        :param save_as:Name of the output (name or path. The ending determines the format).
        :param show:True if a preview is to be made, otherwise False.
        :return:The generated image section (np array).
        """
        box = np.asarray(box)

        ## (1) Crop the image by copying an area of the image
        rect = cv2.boundingRect(box)
        x, y, w, h = rect

        ## It is possible that the x or y value is less than or equal to 0. This leads to
        ## an error in the further process. Therefore, the smallest value is set to 1.
        if x < 0:
            x = 1
        if y < 0:
            y = 1

        croped = image[y:y + h, x:x + w].copy()

        ## (2) make mask
        box = box - box.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [box], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) New image by bitwise AND comparison of mask and cropped image
        new_image = cv2.bitwise_and(croped, croped, mask=mask)

        ## (4) Add a white background by bitwise NOT comparison
        bg = np.ones_like(croped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        new_image = bg + new_image

        ## (5) Use gray image, if desired
        if greyscale == True:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        ## (6) Display image via Open CV if desired
        if show == True:
            BoundingBoxImageHandler.show_image(new_image)

        ## (7) Save image via Open CV, if desired
        if save == True:
            BoundingBoxImageHandler.serialize_image(new_image, save_as)

        return new_image

    @staticmethod
    def put_text(image, box, text, pos_annotation_constants):
        """Inserts the passed text into the passed image. The box defines the area over
        which the text is to be entered.

        All special characters are converted as follows before output:
            ö => oe
            ü => ue
            a => ae
            Ö => Oe
            Ü => Ue
            Ä => Ae

        :param image:The image to be written to.
        :param box:The box that marks the area.
        :param text:The text to be written.
        :param pos_annotation_constants:An object that defines the text output (color, size,...).
        :return:The transfered image with the drawn text
        """
        rect = cv2.boundingRect(box)
        x, y, w, h = rect

        text = str(text).replace("ö", "oe")
        text = str(text).replace("ü", "ue")
        text = str(text).replace("ä", "ae")

        text = str(text).replace("Ö", "Oe")
        text = str(text).replace("Ü", "Ue")
        text = str(text).replace("Ä", "Ae")

        (label_width, label_height), baseline = cv2.getTextSize(text, pos_annotation_constants.FONT(),
                                                                pos_annotation_constants.SCALE(),
                                                                pos_annotation_constants.THICKNESS())

        cv2.putText(image[:, :, ::-1], text, (x, y - 3), pos_annotation_constants.FONT(),
                    pos_annotation_constants.SCALE(),
                    pos_annotation_constants.COLOR(),
                    pos_annotation_constants.THICKNESS())

        return image

    @staticmethod
    def put_border(image, box, rgb_color, thickness=1):
        """Draws a polygon into the transferred image. Most of the cases will be rectangles.

         box = [[a,b],[c,d],[e,f],[g,h]]

        :param image:The image in which you want to draw.
        :param box:A box that defines the polygon. box = [[a,b],[c,d],[e,f],[g,h]].
        :param rgb_color:The color of the frame as RGB value.
        :param thickness:The width of the border.
        :return:The transferred image with the drawn frames.
        """
        r, g, b = rgb_color
        cv2.polylines(image[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                      color=(b, g, r), thickness=thickness)  # cv arbeitet mit BGR!

        return image

    @staticmethod
    def serialize_image(image, path):
        """Saves the transferred image to the location specified in the path.

        :param image:The image to save.
        :param path:The location of the image.
        """
        if image is not None and os.path.exists(os.path.dirname(path)):
            cv2.imwrite(path, image)

    @staticmethod
    def show_image(image):
        """The image to display. After pressing a key, the preview is closed.

        :param image:The image to display.
        """
        if image is not None:
            cv2.imshow("Preview", image)
            cv2.waitKey()

    @staticmethod
    def separate_boxes(path):
        """A method that creates a list of corner coordinates from a text file with a list
        of X and Y values:

        a b c d e f g h => [[a,b],[c,d],[e,f],[g,h]]

        :param path:The path of a text file with X and Y values.
        :return:A box in the format: [[a,b],[c,d],[e,f],[g,h]]
        """
        if os.path.isfile(path):
            file = open(path, mode='r')

            boxes = []
            for line in file:
                line_boxes = []

                if (line.strip() != ''):
                    coords = [int(item) for item in line.split(',') if item.split() != '']

                    new_box = []
                    i = 0
                    for coord in coords:
                        if i == 0 or i == 1:
                            new_box.append(coord)
                            i = i + 1

                        if i == 2:
                            line_boxes.append(new_box)

                            new_box = []
                            i = 0

                    boxes.append(line_boxes)
                    line_boxes = []

            return boxes
        else:
            return []

    @staticmethod
    def BGR_to_RGB(BGR):
        """Converts a RGB color passed as a tuple from BGR (Open CV) to RGB format.

        :param BGR:Color in BGR format (Open CV)
        :return:The color in RGB format
        """
        return (BGR[2], BGR[1], BGR[0])

    @staticmethod
    def RGB_to_BGR(RGB):
        """Converts a RGB color passed as a tuple from RGB to BGR format (Open CV).

        :param RGB:Color in RGB format
        :return:The color in BGR format
        """
        return (RGB[2], RGB[1], RGB[0])
