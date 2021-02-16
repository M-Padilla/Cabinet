"""
Created on Sat Oct 10 15:25:16 2020
Modified on Fri Feb 12 02:56 2020
@author: M.Padilla
"""

import cv2 as cv
import filetype
import json
import numpy as np
import pdf2image
import pytesseract
import getpass
import os
import sys


def box_construction(xywh_array):
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
    else:  # Check is PDF is password protected.
        password = None
        while True:
            try:  # Convert to .jpeg image file
                images_from_path = pdf2image.convert_from_path(
                    file, fmt="jpeg", dpi=400,
                    single_file=True, userpw=password)
                break
            except pdf2image.exceptions.PDFPageCountError:
                print('LOCKED PDF ERROR: PDF is password protected.')
                password = getpass.getpass(
                    prompt='Please provide a valid password: ')
    return images_from_path


def img_cropper(image_np_array, insurer):
    '''
    Crops image to a delimited box to remove footer and blank spaces in the top and sides.
    '''

    # Cropbox's original boundaries (X0,Y0,XF,YF)
    cropbox = (200, -1780, 200, -200)
    # offsets.json contains offset coordinates (+X,+Y) dictionary for cropbox, customized for each insurer.
    with open('offsets.json', mode='r') as offsets_f:  # CHANGE NAME OF FILE LATER OR SWITCH TO MAIN CODE AS GLOBAL VARIABLE?
        # Get offset for the specific insurer's form
        offset = json.load(offsets_f)['offsets'].get(insurer, (0, 0))

    # Adjust cropbox boundaries
    cropbox = (
        cropbox[0] + offset[1], cropbox[1] + offset[1],
        cropbox[2] + offset[0], cropbox[3] + offset[0])
    return image_np_array[cropbox[0]:cropbox[1], cropbox[2]:cropbox[3]]  # (y:y+h, x:x+w)


def img_preprocessor(image_np_array):
    # Convert RGB to BGR (required by OpenCV) gray scale.
    image_np_array = cv.cvtColor(image_np_array, cv.COLOR_BGR2GRAY)
    # Threshold evaluation: If pixel value is greater than 180, pixel value becomes 255 (white).
    ret, image_np_array = cv.threshold(image_np_array, 130, 255, cv.THRESH_BINARY)
    # Blur correction, to remove high frequency content (eg: noise, edges) from the image
    image_np_array = cv.GaussianBlur(image_np_array, (7, 7), sigmaX=0, sigmaY=0)
    return image_np_array


def insurer_identifier(image_np_array):  # WILL BE IMPLEMENTED
    # TO-DO: PATTERN MATCHING OR OCR FOR INSURER'S LOGO
    return "MUNDIAL"


# Visualize image - WILL BE DELETED
def test():
    cv.namedWindow('dmc', cv.WINDOW_NORMAL)
    cv.resizeWindow('dmc', 900, 800)
    cv.imshow('dmc', drawing)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)


file = "C:/Users/user/Documents/Proyectos/Cabinet 1/SOAT Mundial.pdf"
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

insurer = insurer_identifier(image)
# Crop image to a delimited box to remove footer and blank spaces in the top and sides
image = img_cropper(image, insurer)
image = img_preprocessor(image)
cv.imwrite('ASTRALDO.jpeg', image)  # MERELY FOR TEST PURPOSES, WILL BE DELETED

# Retrieve box coordinates (x, y, w, h) for every field.
with open('fields.json', mode='r') as fields_f:
    field_boxes = json.load(fields_f)['field_boxes']

output = {}
for field in field_boxes.keys():
    field_coords = np.array(field_boxes[field]['coords'].get(insurer, field_boxes[field]['coords']['default']))
    # Turns an array of coordinates (x,y,w,h) into an array of coordinates (x,y,x+w,y+h)
    field_coords = box_construction(field_coords)
    # Retrieve image corresponding to field coordinates in the following order: (y,y+h,x,x+w)
    # Then apply OCR and convert to string
    output[field] = str(pytesseract.image_to_string(
        image[field_coords[1]: field_coords[3], field_coords[0]: field_coords[2]],
        config=field_boxes[field]['ocr_config_mode'])).upper()


'''
fromCenter = False
cv.namedWindow('Image',cv.WINDOW_NORMAL)
cv.resizeWindow('Image', 900,800)
r = cv.selectROIs("Image", open_cv_image, fromCenter)
cv.destroyWindow('Image')

for image in images_from_path:
    image.close()
'''
