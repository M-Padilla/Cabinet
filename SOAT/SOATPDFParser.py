"""
Created on Sat Oct 10 15:25:16 2020
Modified on Sun Mar 13 00:09 2021
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
import sys


def box_construction(xywh_array):
    '''
    Turns an array of coordinates (x, y, w, h) into an array of coordinates (x, y, x+w, y+h)
    '''
    xywh_array[2] += xywh_array[0]
    xywh_array[3] += xywh_array[1]
    return xywh_array


def file_checker(pdf_doc):
    if filetype.guess(pdf_doc).extension != 'pdf':
        # Check if file is PDF
        raise TypeError
    elif os.stat(pdf_doc).st_size / (1024 * 1024) > 10.0:
        # Check for PDF size, max 10 MB.
        raise MemoryError
    else:
        # Check is PDF is password protected.
        password = None
        while True:
            try: 
                # Convert to .jpeg image file
                images_from_path = pdf2image.convert_from_path(
                        file, fmt="jpeg", dpi=400,
                        single_file=True, userpw=password)
                break
            except pdf2image.exceptions.PDFPageCountError:
                # Prompt for password
                print('LOCKED PDF ERROR: PDF is password protected.')
                password = getpass.getpass(
                        prompt='Please provide a valid password: ')
    return images_from_path
 

def img_cropper(image_np_array, _insurer):
    '''
    Crops image to a delimited box to remove footer and blank spaces in the top and sides.
    '''
    # Cropbox's original boundaries (X0, Y0, XF, YF)
    cropbox = (200, -1780, 200, -200)
    # Retrieves offset coordinates (+X, +Y) for the cropbox, customized for each insurer.
    # using the global variable offsets
    offset = all_offsets.get(_insurer, (0,0))    
    # Adjust cropbox boundaries
    cropbox = (
            cropbox[0]+offset[1], cropbox[1]+offset[1],
            cropbox[2]+offset[0], cropbox[3]+offset[0])
    return image_np_array[cropbox[0]:cropbox[1], cropbox[2]:cropbox[3]] # (y:y+h, x:x+w)


def img_preprocessor(image_np_array, _insurer=None):
    # Convert RGB to BGR (required by OpenCV) gray scale.
    image_np_array = cv.cvtColor(image_np_array, cv.COLOR_BGR2GRAY)
    if _insurer != None:
        # Threshold evaluation: If pixel value is greater than 180, pixel value becomes 255 (white).
        ret, image_np_array = cv.threshold(image_np_array, 130, 255, cv.THRESH_BINARY)
        # Blur correction, to remove high frequency content (eg: noise, edges) from the image
        image_np_array = cv.GaussianBlur(image_np_array, (7, 7), sigmaX=0, sigmaY=0)
    return image_np_array


def field_img_generator(_image, _field, _field_boxes, _insurer):
    field_coords = np.array(_field_boxes[_field]['coords'].get(_insurer, _field_boxes[_field]['coords']['default']))
    # Turns an array of coordinates (x, y, w, h) into an array of coordinates (x, y, x+w, y+h)
    field_coords = box_construction(field_coords) 
    # Retrieve field image corresponding to field coordinates in the following order: (y:y+h, x:x+w)
    # Then apply OCR and convert to string
    field_image = _image[field_coords[1]: field_coords[3], field_coords[0]: field_coords[2]]
    field_image = img_preprocessor(field_image, _insurer)
    return field_image


def field_ocr(_image, psm_config_modes, _insurer):  
    if _insurer != None: # Standard field recognition
        return str(pytesseract.image_to_string(_image, config=psm_config_modes)).strip().upper()
    else: # Insurer name recognition        
        for mode in psm_config_modes: # Try to identify the insurer using several psm modes
            insurer_txt = str(pytesseract.image_to_string(_image, config=mode)).strip().upper()
            for available_insurer in insurers_list:
                # Checks the available insurers for this policy line using global variable insurers_list
                if available_insurer in insurer_txt:
                    return available_insurer # Implictly returns None if insurer is not found.
    

file="C:/Users/user/Documents/Proyectos/Cabinet 1/SOAT Sura 3.pdf"
pytesseract.pytesseract.tesseract_cmd = 'D:/Programas/anaconda3/pkgs/tesseract-4.1.1-h1fd39ab_3/Library/bin/tesseract.exe'

# Attempt conversion to a NumPy array with RGB values for reading through OpenCV
try:
    image = np.array(file_checker(file)[0])
except Exception as err:
    if err == TypeError:
        print('FILE TYPE ERROR: File is not a PDF or is corrupted.')
    elif err == MemoryError:
        print('SIZE ERROR: Maximum file size is 10 MB')
    sys.exit()

# Retrieve box coordinates (x, y, w, h) for every field.
with open('fields.json', mode='r') as fields_f:
    data = json.load(fields_f)
    field_boxes = data["field_boxes"] # Load boxes, psm mode and regex patterns for every field
    insurers_list = data["available_line_insurers"] # Load available insurers for this policy line.
    all_offsets = data["offsets"] # Load necessary offsets for certain insurers
    del data

cv.imwrite('ASTRALDO.jpeg', image) # MERELY FOR TEST PURPOSES, WILL BE DELETED

# Insurer identification via OCR
insurer, output = None, {}
output['asegurador'] = field_ocr(
        field_img_generator(image, 'asegurador', field_boxes, insurer),
        field_boxes['asegurador']['ocr_config_mode'],
        insurer)
insurer = output['asegurador']

# Crop image to a delimited box to remove footer and blank spaces in the top and sides
image = img_cropper(image, insurer)

# Once the insurer has been identified and the image has been cropped, recognize the remaining fields
for field in field_boxes.keys():
    if field != 'asegurador':
        output[field] = field_ocr(
                field_img_generator(image, field, field_boxes, insurer),
                field_boxes[field]['ocr_config_mode'], insurer)
    else:
        continue
