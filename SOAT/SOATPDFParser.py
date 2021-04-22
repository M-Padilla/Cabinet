"""
Created on Sat Oct 10 15:25:16 2020

@author: M.Padilla
"""


import cv2 as cv
import filetype
import getpass
import json
import numpy as np
import pdf2image
import pytesseract
import os
import re
import sys
import warnings


def build_region_box(xywh_array):
    '''
    Turns an array of coordinates (x, y, w, h) into an array of coordinates
    (x, y, x+w, y+h)
    '''
    x, y, w, h = xywh_array
    return [x, y, x + w, y + h]


def load_config(config_file):
    with open(config_file, mode='r') as config_f:
        data = json.load(config_f)
        # Load boxes, psm mode and regex patterns for every field
        _field_boxes = data["field_boxes"]
        # Load available insurers for this policy line.
        _insurers_list = data["supported_line_insurers"]
        # Load necessary offsets for certain insurers
        _all_offsets = data["offsets"]
    return _field_boxes, _insurers_list, _all_offsets


def check_input_file(pdf_doc):
    # Check if file is PDF
    if filetype.guess(pdf_doc).extension != 'pdf':
        raise TypeError
    # Check for PDF size, max 10 MB.
    if os.stat(pdf_doc).st_size / (1024 * 1024) > 10.0:
        raise MemoryError
    # Check is PDF is password protected.
    password = None
    while True:
        try:
            # Convert to .jpeg image file
            images_from_path = pdf2image.convert_from_path(
                    file, fmt="jpeg", dpi=400, single_file=True,
                    userpw=password)
            break
        except pdf2image.exceptions.PDFPageCountError:
            # Prompt for password
            print('LOCKED PDF ERROR: PDF is password protected.')
            password = getpass.getpass(
                    prompt='Please provide a valid password: ')
    return images_from_path


def crop_image(image_np_array, _insurer):
    '''
    Crops image to a delimited box to remove footer and blank spaces in the top
    and sides.
    '''
    # Cropbox's original boundaries (X0, Y0, XF, YF)
    cropbox = (200, -1780, 200, -200)
    # Retrieves offset coordinates (+X, +Y) for the cropbox, customized for
    # each insurer using the global variable offsets.
    offset = all_offsets.get(_insurer, (0, 0))
    # Adjust cropbox boundaries
    cropbox = (cropbox[0]+offset[1], cropbox[1]+offset[1],
               cropbox[2]+offset[0], cropbox[3]+offset[0])
    return image_np_array[cropbox[0]:cropbox[1], cropbox[2]:cropbox[3]]
    # (y:y+h, x:x+w)


def preprocess_image(image_np_array, _insurer=None):
    # Convert RGB to BGR (required by OpenCV) gray scale.
    image_np_array = cv.cvtColor(image_np_array, cv.COLOR_BGR2GRAY)
    if _insurer is not None:
        # Threshold evaluation: If pixel value is greater than 180, pixel value
        # becomes 255 (white).
        ret, image_np_array = cv.threshold(image_np_array, 130, 255,
                                           cv.THRESH_BINARY)
        # Blur correction, to remove high frequency content (eg: noise, edges)
        # from the image
        image_np_array = cv.GaussianBlur(image_np_array, (7, 7),
                                         sigmaX=1, sigmaY=2)
    return image_np_array


def create_field_image(_image, _field, _field_boxes, _insurer):
    field_coords = np.array(_field_boxes[_field]['coords'].get(
                            _insurer,
                            _field_boxes[_field]['coords']['default']))
    # Turns an array of coordinates (x, y, w, h) into an array of coordinates
    # (x, y, x+w, y+h)
    field_coords = build_region_box(field_coords)
    # Retrieve field image corresponding to field coordinates in the following
    # order: (y:y+h, x:x+w), then apply OCR and convert to string
    field_image = _image[field_coords[1]: field_coords[3],
                         field_coords[0]: field_coords[2]]
    field_image = preprocess_image(field_image, _insurer)
    return field_image


def lint_field(field_string, mode):

    def date_linter(date_string):
        date_string = re.sub(r'[^\d\s]', '', date_string).strip()
        if check_ocr_text_w_regex(r'\d{5,6}\s\d{1,2}',
                                  date_string) is not None:
            date_string = date_string[:4] + ' ' + date_string[4:]
        return date_string

    def empty_linter(field_string):
        if field_string == 'OB':
            return ''
        return field_string

    def numeric_string_linter(numeric_string):
        '''
        Performs a series of validations and corrections for a number stored as
        a string.
        '''
        numeric_string = re.sub(r'[^\d\.\,]', '', numeric_string).strip()

        if check_ocr_text_w_regex(r'\d{1,3}([\.,]?\d{3})*[\.,][0]{2}$',
                                  numeric_string) is not None:
            # Remove decimal part when it is composed by a double zero.
            numeric_string = numeric_string[:-3]

        if check_ocr_text_w_regex(r'^\d{1,3}([\.,]\d{3})+$',
                                  numeric_string) is not None:
            # Remove nondecimal separators when the string is a whole number.
            numeric_string = re.sub("[\.,]", "", numeric_string)

        if check_ocr_text_w_regex(r'^\d{1,3}([\.,]\d{3})+[\.,]\d{1,2}$',
                                  numeric_string) is not None:
            # Remove nondecimal separators and correct decimal separator when
            # the string has a valid decimal part.
            if numeric_string[-3] in (',', '.'):
                decimal_sep = numeric_string[-3]
            else:
                decimal_sep = numeric_string[-2]

            nondecimal_sep = ',' if decimal_sep == '.' else '.'
            numeric_string = numeric_string.translate(
                    str.maketrans(decimal_sep, ".", nondecimal_sep))

        return numeric_string

    def placa_linter(placa_string):
        # Delete the & char in the placa field, sometimes mistakenly duplicated
        # by pytesseract when the char '8' is present.
        placa_string = placa_string.replace("&", "")
        if len(placa_string) == 6:
            # If the placa field belongs to a motorcycle or car, replace
            # 'O' by '0' in certain positions of the string
            if output['clase_veh'] == 'MOTOCICLETA':
                # The placa field belongs to a motorcycle
                placa_string = ''.join([placa_string[0:3],
                                        placa_string[3:-1].replace("O", "0"),
                                        placa_string[-1]])
            else:
                # The placa field belongs to a car
                placa_string = ''.join([placa_string[0:3],
                                        placa_string[3:].replace("O", "0")])
        return placa_string

    linter_functions_dict = {"date": date_linter, "empty": empty_linter,
                   "numeric_string": numeric_string_linter,
                   "placa": placa_linter}
    function = linter_functions_dict[mode]

    return function(field_string)


def ocr_field(_image, psm_config_modes, _insurer):
    if _insurer is not None:  # Standard field recognition
        return str(pytesseract.image_to_string(_image, config=psm_config_modes)
                   ).translate(str.maketrans('\n|', '  ')).strip().upper()
    # Insurer name recognition
    for mode in psm_config_modes:
        # Try to identify the insurer using several psm modes
        insurer_txt = str(pytesseract.image_to_string(_image, config=mode)
                          ).strip().upper()
        for supported_insurer in insurers_list:
            # Checks the available insurers for this policy line using
            # global variable insurers_list
            if supported_insurer in insurer_txt:
                # Implicitly returns None if insurer is not found.
                return supported_insurer
    warnings.warn("WARNING: Insurer is not currently supported. Default "
                  "parameters will be used. This may affect recognition "
                  "accuracy.")


def check_ocr_text_w_regex(_regex_pattern, text):
    # Checks if a text follows a regex pattern; if true, will return the
    # text, otherwise returns None.
    _matched_text = re.search(_regex_pattern, text)
    if _matched_text is not None:
        return _matched_text.group()


file = r"C:/Users/user/Documents/Proyectos/Cabinet 1/SOAT Mundial.pdf"
pytesseract.pytesseract.tesseract_cmd = r'D:/Programas/anaconda3/pkgs/'\
                                        'tesseract-4.1.1-h1fd39ab_3/Library/'\
                                        'bin/tesseract.exe'

# Attempt conversion to NumPy array with RGB values for reading through OpenCV
try:
    image = np.array(check_input_file(file)[0])
except Exception as err:
    if err == TypeError:
        print('FILE TYPE ERROR: File is not a PDF or is corrupted.')
    elif err == MemoryError:
        print('SIZE ERROR: Maximum file size is 10 MB')
    sys.exit()

field_boxes, insurers_list, all_offsets = load_config('config.json')

# Insurer identification via OCR
insurer, output = None, {}
output['asegurador'] = ocr_field(
        create_field_image(image, 'asegurador', field_boxes, insurer),
        field_boxes['asegurador']['ocr_config_mode'],
        insurer)
insurer = output['asegurador']

# Crop image to a delimited box to remove footer and blank spaces in the top
# and sides
image = crop_image(image, insurer)
# Fields that do not follow their designated regex pattern
needs_review_fields = []

# Once the insurer has been identified and the image has been cropped,
# recognize the remaining fields
for field in field_boxes.keys():
    if field == 'asegurador':
        continue
    # Field OCR
    output[field] = ocr_field(
            create_field_image(image, field, field_boxes, insurer),
            field_boxes[field]['ocr_config_mode'], insurer)
    try:
        # Attempt to validate the recognized field according to its
        # designated regex pattern
        matched_text = check_ocr_text_w_regex(
                                    field_boxes[field]['regex_pattern'],
                                    output[field])
        if matched_text is not None:  # Successful validation
            output[field] = matched_text
        else:  # Failed validation
            needs_review_fields.append(field)
    except:
        pass

corrected_fields = []  # Reviewed fields that were corrected.

# Perform some validations and corrections for recognized fields in need of
# review.
for field in needs_review_fields:
    output[field] = lint_field(output[field], "empty")
    if field in ('placa'):
        output[field] = lint_field(output[field], "placa")
    elif field in ('fecha_expedicion', 'fecha_inicio', 'fecha_vencimiento'):
        output[field] = lint_field(output[field], "date")
    elif field in ('cap_ton', 'cilindraje-vatios', 'contrib_fosyga',
                   'prima_soat', 'tasa_runt', 'total_a_pagar'):
        output[field] = lint_field(output[field], "numeric_string")
    # Check if the field now complies with its designated regex pattern
    if check_ocr_text_w_regex(field_boxes[field]['regex_pattern'],
                              output[field]) is not None:
        corrected_fields.append(field)


# Retain the fields that weren't corrected for manual correction from the user.
needs_review_fields = list(
                        set(needs_review_fields).difference(corrected_fields))
del corrected_fields
