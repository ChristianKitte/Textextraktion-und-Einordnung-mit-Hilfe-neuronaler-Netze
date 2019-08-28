""" Function of evaluation.py
The script makes it possible to quickly and easily perform all stored functions for testing the
system. The results are stored in the defined directories.

The functions can be executed immediately. All required images are stored. In addition, the script
can be adapted for further images.
"""
import os
import shutil

import cv2

import constant
from scanner import Scanner


def evaluate(evallist, scanner, basedir):
    """The function receives and evaluates a list of image files. The resulting image is stored in
    the subdirectory passed with (to the base directory).

    All predicted text areas as well as the predicted text are simply entered into the output image.

    :param evallist:A list with file names and the name of the output directory.
    :param scanner:An instance of the class Scanner.
    :param basedir:The parent directory to the passed directories (this must exist).
    """
    for file, dir in evallist:
        cur_dir = os.path.join(basedir, dir)

        if os.path.exists(cur_dir):
            shutil.rmtree(cur_dir)

        os.mkdir(cur_dir)

        img_in = cv2.imread(os.path.join(basedir, file))[:, :, ::-1]

        if img_in is not None:
            img_out = scanner.scann(img=img_in, evaluation_mode=True)
            cv2.imwrite(os.path.join(cur_dir, '001.jpg'), img_out)

            print('Finished')
        else:
            print('Image not readable')


def evaluate_char(versions, count_from, count_to, zeros, scanner, basedir, format='.jpg'):
    """A routine for evaluating the text recognizer. Images with texts are read in and output.

    A list of images with the following formatting is expected in the transferred directory:

        001_Version1.jpg
        001_Version2.jpg
        002_Version1.jpg
        002_Version2.jpg

    The versions are transferred in the Versions list:
    ['_Version1.jpg', '_Version2.jpg') etc. The numbers are taken from count_from, count_to (here: count_from=1,
    count_to=2, zeros=3).

    The number of versions per image must always be the same !

    :param versions:The available image versions
    :param count_from:Die Startbereich der Zahlen für die Benamung. (inklusive)
    :param count_to:The end range of the numbers for the naming. (inclusive)
    :param zeros:Number of leading zeros
    :param scanner:An instance of the class Scanner.
    :param basedir:The parent directory to the passed directories (this must exist).
    :param format:The output format in the form of a valid file extension (e.g. .jpg). Supports all formats
    supported by Open CV
    """
    for i in range(count_from, count_to + 1, 1):
        i_str = str(i).zfill(zeros)
        cur_out_dir = os.path.join(basedir, i_str)

        if os.path.exists(cur_out_dir):
            shutil.rmtree(cur_out_dir)

        os.mkdir(cur_out_dir)

        for name_x in versions:
            img_name = i_str + name_x
            img_in = cv2.imread(os.path.join(basedir, img_name))[:, :, ::-1]

            version = name_x[:-4]

            if img_in is not None:
                predicted_text = scanner.predict_text(img=img_in)
                cv2.imwrite(os.path.join(cur_out_dir, i_str + '_' + version + '_' + predicted_text + format), img_in)

                print('Finished')
            else:
                print('Image not readable')


def evaluate_database(keywords, scanner):
    """Scrolls through the passed list of keywords and queries the database for each keyword.
    The result is displayed directly on the console.

    :param keywords:A list with keywords
    :param scanner:An instance of the class Scanner.
    """
    for term in keywords:
        state, id, ident = scanner.db_contains(term)
        print(term + ' evaluate to ' + str(state) + ' stands for ' + ident)


if __name__ == '__main__':
    """Is executed when the file is executed directly. It executes a series of defined actions to evaluate 
    the system on the basis of the stored images.

    For this purpose, a text detection and/or a text recognition is carried out and the results are stored 
    in the form of images.
    
    Methodenaufrufe müssen zunächst frei geschaltet werden. Hierfür das Hashtag entfernen.
    """
    ## A list of defined images with normal text for evaluation purposes
    evallist_text = [['text_black_sharp.jpg', 'text_black_sharp'],
                     ['text_black_sharp_blur.jpg', 'text_black_sharp_blur'],
                     ['text_forest_black_sharp.jpg', 'text_forest_black_sharp'],
                     ['text_forest_black_sharp_blur.jpg', 'text_forest_black_sharp_blur']]

    ## A list of defined images with special chars for evaluation purposes
    evallist_special_text = [['special_black_sharp.jpg', 'special_black_sharp'],
                             ['special_black_sharp_blur.jpg', 'special_black_sharp_blur'],
                             ['special_forest_black_sharp.jpg', 'special_forest_black_sharp'],
                             ['special_forest_black_sharp_blur.jpg', 'special_forest_black_sharp_blur']]

    ## A list with defined examples for demonstration purposes
    samplelist = [['001.jpg', '001jpg'],
                  ['002.jpg', '002jpg'],
                  ['003.jpg', '003jpg'],
                  ['001.png', '001png'],
                  ['002.png', '002png']]

    ## A list of predefined name modules for versions that are required as parameters for evaluate_char.
    versions = ['_sharp.jpg', '_forrest_sharp.jpg', '_sharp_blur.jpg', '_forrest_sharp_blur.jpg']

    keywords = ['E129', 'E 129', 'E  129', 'e129', 'e 129', 'e  129', 'E-129', 'E - 129', 'e-129', 'e - 129',
                'E160e', 'E 160e', 'E  160e', 'e160e', 'e 160e', 'e  160e', 'E-160e', 'E - 160e', 'e-160e', 'e - 160e',
                'E160 e', 'E 160 e', 'E  160 e', 'e160 e', 'e 160 e', 'e  160 e', 'E-160 e', 'E - 160 e', 'e-160 e',
                'e - 160 e',
                'Gelborange S', 'GelbOrange S', 'Gelborange s', 'GelbOrange s',
                'Natriumdiacetat', 'NatriumDiacetat',
                'Trinatriumcitrat', 'TrinatrIumcitrat',
                'Gelborange', 'GelbOrange']

    scanner = Scanner(refresh_db=True)
    const = constant

    ## text - detection and recognition whole picture
    # evaluate(evallist_text, scanner, os.path.join(const.EVALUATION_DIR, 'text'))
    # evaluate(evallist_special_text, scanner, os.path.join(const.EVALUATION_DIR, 'special_text'))

    ## samples - detection and recognition whole picture
    # evaluate(samplelist, scanner, const.SAMPLE_DIR)

    ## chars - recognition whole picture
    # evaluate_char(versions, 1, 12, 3, scanner, os.path.join(const.EVALUATION_DIR, 'chars'))
    # evaluate_char(versions, 1, 7, 3, scanner, os.path.join(const.EVALUATION_DIR, 'special_chars'))

    ## Database - Find specific keywords. The terms can only be evaluated as True, if certain preprocessings
    ## have been done and certain columns have been searched. TRUE must be returned for all search words!
    evaluate_database(keywords, scanner)
