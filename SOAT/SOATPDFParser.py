"""
Created on Sat Oct 10 15:25:16 2020

@author: M.Padilla
"""

import ast
import cv2 as cv
import filetype
import numpy as np
import pdf2image
import pytesseract
import os


def box_construction(fields_array):
    #Turns an array of coordinates (x,y,w,h) into an array of coordinates (x,y,x+w,y+h)
    fields_array[:,2]+= fields_array[:,0]
    fields_array[:,3]+= fields_array[:,1]
    return fields_array
    
#Visualize image
def test():
    cv.namedWindow('dmc',cv.WINDOW_NORMAL)
    cv.resizeWindow('dmc', 900,800)
    cv.imshow('dmc', drawing)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)


file= "C:/Users/user/Documents/Proyectos/Cabinet 1/SOAT Patricia.pdf"
pytesseract.pytesseract.tesseract_cmd = 'D:/Programas/anaconda3/pkgs/tesseract-4.1.1-h1fd39ab_3/Library/bin/tesseract.exe'


try:
    #Check if file is a PDF
    if filetype.guess(file).extension != 'pdf':
        raise TypeError
    elif os.stat(file).st_size / (1024 * 1024) > 10.0: #Check for file size, max 10 MB.
        raise MemoryError
except pdf2image.PDFPageCountError:
    print('A')
except TypeError:
    print('FILE TYPE ERROR: File is not a PDF or is corrupted.')
except MemoryError:
    print('SIZE ERROR: Maximum file size is 10 MB')
else:
    
    #Offset coordinates (+X,+Y) dictionary for cropbox, customized for each insurer.
    policy_offset = ast.literal_eval(open("offsets.dat", "r").read())
    # Convert to .jpeg image file.
    images_from_path = pdf2image.convert_from_path(file,
                                                   fmt="jpeg",
                                                   dpi=400,
                                                   single_file=True)
    for image in images_from_path:
        #Convert to a NumPy array with RGB values for reading through OpenCV
        open_cv_image = np.array(image)
        #Check if image is coloured and has correct dimensions.
        try:
            if ( 
                    open_cv_image.shape[0] != 4400 or 
                    open_cv_image.shape[1] != 3400 or
                    open_cv_image.shape[2] != 3
                ):
                raise TypeError
        except TypeError:
            print('ERROR: The file is not a standard SOAT document.')
            del open_cv_image
        else:    
            #Crop image to a delimited box to remove footer and blank spaces in the top and sides (y:y+h, x:x+w)
            cropbox= (200,-1780, 200, -200) #Crop box's original boundaries (X0,Y0,XF,YF)
            insurer_name = "MUNDIAL" #TO-DO: Extend to dilate for Sura, to compensate for thinner font.
            offset= policy_offset.get(insurer_name,(0,0)) #Get offset for the specific insurer's form
            cropbox= (cropbox[0]+offset[1], cropbox[1]+offset[1], cropbox[2]+offset[0], cropbox[3]+offset[0]) #Offset cropbox boundaries
            open_cv_image= open_cv_image[cropbox[0]:cropbox[1], cropbox[2]:cropbox[3]] #Crop image
            # Convert RGB to BGR (required by OpenCV) gray scale.
            open_cv_image = cv.cvtColor(open_cv_image,cv.COLOR_BGR2GRAY)     
            #Threshold evaluation: If pixel value is greater than 180, pixel value becomes 255 (white).
            ret,open_cv_image = cv.threshold(open_cv_image,130,255,cv.THRESH_BINARY)
            #Blur correction, to remove high frequency content (eg: noise, edges) from the image
            open_cv_image = cv.GaussianBlur(open_cv_image, (7,7), sigmaX=0, sigmaY=0)  
            cv.imwrite('ASTRALDO.jpeg',open_cv_image)               
                        
            
            #Numpy array containing all regions of interest for fields.
            #(x,y,w,h)  
            field_boxes= np.loadtxt('field_boxes.dat',delimiter=',').astype(int)
            
            if insurer_name == "SURA":
                #Modify coordinates of the total_premium field if SOAT document is issued by Sura.
                field_boxes[28] = np.array([295,2070,445,70])
            
            #Dictionary containing the row indexes in the 'fields' numpy array belonging to each field.
            field_dict = {"body": 0,
                        "chassis_serial": 1,
                        "coverage": 2, #Per victim
                        "coverage_limits": 3, #Limits given as number of daily minimum legal wages.
                        "cylinder_cap": 4,
                        "expiration_date": 5, #All policies expire at 23:59 of the given day.
                        "fosyga_fee": 6,
                        "inception_date": 7,
                        "insurer": 8,
                        "insurer_code": 9,
                        "issue_date": 10, #All policies get issued at 00:00 of the given day.
                        "line": 11,
                        "manufacturer": 12,
                        "model_year": 13,
                        "motor_serial": 14,
                        "passenger_cap": 15,
                        "plate": 16,
                        "policy_number": 17,
                        "policyholder_city": 18,
                        "policyholder_fullname": 19,
                        "policyholder_id": 20,
                        "policyholder_id_class": 21,
                        "policyholder_phone": 22,
                        "pricing_rate_code": 23,
                        "runt_fee": 24,
                        "service_class": 25,
                        "soat_premium": 26,
                        "ton_cap": 27,
                        "total_premium": 28,
                        "vehicle_class": 29,
                        "vin": 30}
                       
            #Calculate absolute coordinates for field boxes.
            field_boxes = box_construction(field_boxes)
          
            
            
            #OCR
            for key in field_dict.keys():
                field_dict[key] = field_boxes[field_dict[key]]
                img = open_cv_image[field_dict[key][1] : field_dict[key][3],
                                   field_dict[key][0] : field_dict[key][2]]
                field_dict[key] = str(pytesseract.image_to_string(img, config='--psm 11'))
            
            print(field_dict)
            
                  
             
            
        for image in images_from_path:
            image.close()

