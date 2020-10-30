# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:25:16 2020

@author: user
"""

from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2 as cv
import tempfile


#File name
'''
filename= "SOAT Sura 2.pdf" 
fd = open(filename, "rb")

pages = convert_from_path("C:/Users/user/Documents/Proyectos/Cabinet 1/SOAT Bolivar.pdf",
                          dpi=150,
                          fmt="png", 
                          output_folder="C:/Users/user/Documents/Proyectos/Cabinet 1",
                          last_page=1)
'''

#Visualize image
def test():
    cv.namedWindow('dmc',cv.WINDOW_NORMAL)
    cv.resizeWindow('dmc', 900,800)
    cv.imshow('dmc', drawing)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)


# Convert to .jpeg image file using a temporary file.
with tempfile.TemporaryDirectory() as path:
    images_from_path = convert_from_path("C:/Users/user/Documents/Proyectos/Cabinet 1/SOAT CARDENAS.pdf",
                                         fmt="jpeg",
                                         dpi=400,
                                         output_folder=path,
                                         single_file=True)
    
    for image in images_from_path:
        #Convert to a NumPy array with RGB values for reading through OpenCV
        open_cv_image= np.array(image)
        #Crop image to remove footer and blank spaces in the top and sides (y:y+h, x:x+w)
        open_cv_image= open_cv_image[100:-1800,200:-200] 
        open_cv_image = cv.bitwise_not(open_cv_image)
        im= open_cv_image.copy()
        # Convert RGB to BGR (required by OpenCV) gray scale.
        open_cv_image = cv.cvtColor(open_cv_image,cv.COLOR_BGR2GRAY)
        #Blur correction, to remove high frequency content (eg: noise, edges) from the image
        open_cv_image = cv.GaussianBlur(open_cv_image, (9,9), 0)  
        #Threshold evaluation: If pixel value is greater than 180, pixel value becomes 255 (white).
        ret,open_cv_image = cv.threshold(open_cv_image,130,255,cv.THRESH_BINARY)
        cv.imwrite('CSTRALDO.jpeg',open_cv_image)
        '''     
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (14,2))
        open_cv_image = cv.dilate(open_cv_image, kernel, iterations=4)
        '''
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (15,7))
        open_cv_image = cv.dilate(open_cv_image, kernel, iterations=4)
        cv.imwrite('ASTRALDO.jpeg',open_cv_image)
        open_cv_image= cv.Canny(open_cv_image, 0, 255)
        
        
        _, contours, hierarchy = cv.findContours(open_cv_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros((open_cv_image.shape[0], open_cv_image.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (255, 0, 255)
            cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        
        test()

        
        '''
        #Dictionary containing all regions of interest, cropped from open_cv_image.
        fields = {"body": (2003, 1206,  927,  153),
                "chassis_serial": (940, 1396,  833,  166),
                "coverage": (1703, 2009,  530,  459),
                "coverage_limit": (2296, 1990,  654,  500),
                "cylinder_cap": (2150,  996,  356,  172),
                "expiration_date": (993,  628,  423,  153),
                "fosyga_fee": (733, 2003,  447,  159),
                "inception_date": (553,  606,  413,  169),
                "insurer": (2336,   59,  664,  600),
                "insurer_code": (110, 1796,  540,  163),
                "issue_date": (190,  600,  326,  156),
                "line": (283, 1293, 1703,   75),
                "manufacturer": (283, 1209, 1697, 75),
                "model_year": (2533, 1000,  380,  165),
                "motor_serial": (113, 1409,  793,  153),
                "passenger_cap": (106, 1203,  157,  162),
                "plate": (520, 1006,  323,  162),
                "policy_number": (105,  987,  403,  188),
                "policyholder_city": (2506, 1593,  440,  169),
                "policyholder_fullname": (116, 1587, 1134,  169),
                "policyholder_id": (2083, 1587,  410,  175),
                "policyholder_id_class": (1690, 1587,  373,  169),
                "policyholder_phone": (1263, 1587,  407,  178),
                "pricing_rate_code": (106, 2006,  150,  156),
                "runt_fee": (1190, 2000,  466,  168),
                "service_class": (1476, 1006,  640,  165),
                "soat_premium": (280, 2003,  443,  172),
                "ton_cap": (2670, 1396,  246,  160),
                "total_premium": (83, 2184,  357,  178),
                "vehicle_class": (876, 1006,  570,  159),
                "vin": (1803, 1400,  837,  165)}
        
        key="total_premium"
        cv.imwrite('BSTRALDO.jpeg',open_cv_image[fields[key][1]:fields[key][1]+fields[key][3],
                                                 fields[key][0]:fields[key][0]+fields[key][2]])
        '''
        
        '''
        fromCenter = False
        cv.namedWindow('Image',cv.WINDOW_NORMAL)
        cv.resizeWindow('Image', 900,800)
        r = cv.selectROIs("Image", open_cv_image, fromCenter)
        cv.destroyWindow('Image')
        '''
        
        
        
    for image in images_from_path:
        image.close()

'''
c = line_items_coordinates[10]
img = open_cv_image[c[0][1]:c[1][1], c[0][0]:c[1][0]]
cv.imshow('dmc', img); cv.waitKey(0); cv.destroyAllWindows(); cv.waitKey(1)
'''
