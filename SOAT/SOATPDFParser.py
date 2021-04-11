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


def box_construction(xywh_array):
    # This could be deleted if (x,y,x+w,y+h) is stored directly in the JSON file, in the correct order
    '''
    Turns an array of coordinates (x, y, w, h) into an array of coordinates
    (x, y, x+w, y+h)
    '''
    xywh_array[2] += xywh_array[0]
    xywh_array[3] += xywh_array[1]
    return xywh_array


def config_loader(config_file):
    with open(config_file, mode='r') as config_f:
        data = json.load(config_f)
        # Load boxes, psm mode and regex patterns for every field
        _field_boxes = data["field_boxes"]
        # Load available insurers for this policy line.
        _insurers_list = data["supported_line_insurers"]
        # Load necessary offsets for certain insurers
        _all_offsets = data["offsets"]
    return _field_boxes, _insurers_list, _all_offsets


def file_checker(pdf_doc):
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


def img_cropper(image_np_array, _insurer):
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
    cropbox = (
            cropbox[0]+offset[1], cropbox[1]+offset[1],
            cropbox[2]+offset[0], cropbox[3]+offset[0])
    return image_np_array[cropbox[0]:cropbox[1], cropbox[2]:cropbox[3]]
    # (y:y+h, x:x+w)


def img_preprocessor(image_np_array, _insurer=None):
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


def field_img_generator(_image, _field, _field_boxes, _insurer):
    field_coords = np.array(_field_boxes[_field]['coords'].get(
                            _insurer,
                            _field_boxes[_field]['coords']['default']))
    # Turns an array of coordinates (x, y, w, h) into an array of coordinates
    # (x, y, x+w, y+h)
    field_coords = box_construction(field_coords)  # This could be deleted if (x,y,x+w,y+h) is stored directly in the JSON file, in the correct order
    # Retrieve field image corresponding to field coordinates in the following
    # order: (y:y+h, x:x+w), then apply OCR and convert to string
    field_image = _image[field_coords[1]: field_coords[3],
                         field_coords[0]: field_coords[2]]
    field_image = img_preprocessor(field_image, _insurer)
    return field_image


def field_linter(field_string, mode):
    
    def date_linter(date_string):
        date_string = re.sub('[^\d\s]', '', date_string).strip()
        if regex_text_checker('\d{5,6}\s\d{1,2}',
                              date_string) is not None:
            date_string = date_string[:4] + ' ' + date_string[4:]
        return date_string
    
    def empty_linter(field_string):
        if len(field_string) == 2 and field_string == 'OB':
            return ''
        return field_string
    
    def numeric_string_format_linter(numeric_string):
        '''
        Performs a series of validations and corrections for a number stored as a
        string.
        '''
        numeric_string = re.sub('[^\d\.\,]', '', numeric_string).strip()
        
        if regex_text_checker('\d{1,3}([\.,]?\d{3})*[\.,][0]{2}$',
                              numeric_string) is not None:
            numeric_string = numeric_string[:-3]
    
        if regex_text_checker('^\d{1,3}([\.,]\d{3})+$',
                              numeric_string) is not None:
            numeric_string = re.sub("[\.,]", "", numeric_string)
    
        if regex_text_checker('^\d{1,3}([\.,]\d{3})+[\.,]\d{1,2}$',
                              numeric_string) is not None:
            separators = [numeric_string.rindex('.'), numeric_string.rindex(',')]
            separators.sort()
            nondecimal_separator, decimal_separator = separators
            numeric_string = numeric_string.replace(
                                                    decimal_separator, 'sep'
                                                    ).replace(
                                                    nondecimal_separator, ''
                                                    ).replace('sep', '.')
        
        if regex_text_checker('^\d+,[1-9]0?', numeric_string) is not None:
            numeric_string = numeric_string.replace(",", ".")
    
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
    
    
    function = eval(mode + "_linter")
    return function(field_string)


def field_ocr(_image, psm_config_modes, _insurer):
    if _insurer is not None:  # Standard field recognition
        return str(pytesseract.image_to_string(_image, config=psm_config_modes)
                   ).strip().upper().replace('\n', " ")
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


def regex_text_checker(_regex_pattern, text):
    # Checks if a text follows a regex pattern; if true, will return the
    # text, otherwise returns None.
    _matched_text = re.search(_regex_pattern, text)
    if _matched_text is not None:
        return _matched_text.group()


file = "C:/Users/user/Documents/Proyectos/Cabinet 1/SOAT Patricia.pdf"
pytesseract.pytesseract.tesseract_cmd = 'D:/Programas/anaconda3/pkgs/'\
                                        'tesseract-4.1.1-h1fd39ab_3/Library/'\
                                        'bin/tesseract.exe'

# Attempt conversion to NumPy array with RGB values for reading through OpenCV
try:
    image = np.array(file_checker(file)[0])
except Exception as err:
    if err == TypeError:
        print('FILE TYPE ERROR: File is not a PDF or is corrupted.')
    elif err == MemoryError:
        print('SIZE ERROR: Maximum file size is 10 MB')
    sys.exit()

field_boxes, insurers_list, all_offsets = config_loader('config.json')

# Insurer identification via OCR
insurer, output = None, {}
output['asegurador'] = field_ocr(
        field_img_generator(image, 'asegurador', field_boxes, insurer),
        field_boxes['asegurador']['ocr_config_mode'],
        insurer)
insurer = output['asegurador']

# Crop image to a delimited box to remove footer and blank spaces in the top
# and sides
image = img_cropper(image, insurer)
# Fields that do not follow their designated regex pattern
needs_review_fields = []

# Once the insurer has been identified and the image has been cropped,
# recognize the remaining fields
for field in field_boxes.keys():
    if field == 'asegurador':
        continue
    # Field OCR
    output[field] = field_ocr(
            field_img_generator(image, field, field_boxes, insurer),
            field_boxes[field]['ocr_config_mode'], insurer)
    try:
        # Attempt to validate the recognized field according to its
        # designated regex pattern
        matched_text = regex_text_checker(
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
    output[field] = field_linter(output[field], "empty")
    if field in ('placa'):
        output[field] = field_linter(output[field], "placa")
    elif field in ('fecha_expedicion', 'fecha_inicio', 'fecha_vencimiento'):
        output[field] = field_linter(output[field], "date")
    elif field in ('cap_ton', 'cilindraje-vatios', 'contrib_fosyga',
                   'prima_soat', 'tasa_runt', 'total_a_pagar'):
        output[field] = field_linter(output[field], "numeric_string_format")
    # Check if the field now complies with its designated regex pattern
    if regex_text_checker(field_boxes[field]['regex_pattern'],
                          output[field]) is not None:
        corrected_fields.append(field)


# Retain the fields that weren't corrected for manual correction from the user.
needs_review_fields = list(
                        set(needs_review_fields).difference(corrected_fields))
del corrected_fields
