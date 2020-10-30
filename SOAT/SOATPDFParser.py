"""
Created on Sat Oct 10 15:25:16 2020

@author: M.Padilla
"""

from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2 as cv


def box_construction(fields_array):
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



#Offset coordinates (+X,+Y) dictionary for cropbox, customized for each insurer.
policy_offset={"AXA COLPATRIA": (0,5),
               "BOLIVAR": (0,-5),
               "MUNDIAL": (0,0),
               "SURA": (0,120)}

    

pytesseract.pytesseract.tesseract_cmd = 'D:/Programas/anaconda3/pkgs/tesseract-4.1.1-h1fd39ab_3/Library/bin/tesseract.exe'

# Convert to .jpeg image file.
images_from_path = convert_from_path("C:/Users/user/Documents/Proyectos/Cabinet 1/SOAT CARDENAS.pdf",
                                         fmt="jpeg",
                                         dpi=400,
                                         single_file=True)
    
for image in images_from_path:
    #Convert to a NumPy array with RGB values for reading through OpenCV
    open_cv_image= np.array(image)
    #Crop image to a box delimited by (Y,X) to remove footer and blank spaces in the top and sides (y:y+h, x:x+w)
    cropbox= (200,-1780, 200, -200)
    insurer_name = "SURA" #TO-DO: Extend to dilate for Sura, to compensate for thinner font.
    offset= policy_offset.get(insurer_name,(0,0))
    cropbox= (cropbox[0]+offset[1], cropbox[1]+offset[1], cropbox[2]+offset[0], cropbox[3]+offset[0]) 
    open_cv_image= open_cv_image[cropbox[0]:cropbox[1], cropbox[2]:cropbox[3]]
    # Convert RGB to BGR (required by OpenCV) gray scale.
    open_cv_image = cv.cvtColor(open_cv_image,cv.COLOR_BGR2GRAY)     
    #Threshold evaluation: If pixel value is greater than 180, pixel value becomes 255 (white).
    ret,open_cv_image = cv.threshold(open_cv_image,130,255,cv.THRESH_BINARY)
    #Blur correction, to remove high frequency content (eg: noise, edges) from the image
    open_cv_image = cv.GaussianBlur(open_cv_image, (7,7), sigmaX=0, sigmaY=0)  
    cv.imwrite('ASTRALDO.jpeg',open_cv_image)               
                
    
    #Numpy array containing all regions of interest, cropped from open_cv_image.
    #(x,y,w,h)  
    fields= np.array([[2000,1150,925,120],
                    [940,1350,845,120],
                    [1705,1960,555,435],
                    [2300,1940,220,450],
                    [2145,950,370,120],
                    [1100,570,315,80],
                    [740,1960,440,110],
                    [640,570,325,75],
                    [2465,0,480,530],
                    [110,1740,540,120],
                    [186,570,340,75],
                    [445,1185,1545,85],
                    [445,1100,1545,85],
                    [2530,950,390,120],
                    [110,1350,810,115],
                    [110,1150,160,120],
                    [520,950,335,120],
                    [110,950,370,120],
                    [2515,1540,410,120],
                    [110,1540,1135,120],
                    [2085,1570,420,95],
                    [1690,1570,380,90],
                    [1265,1540,410,120],
                    [105,1950,150,120],
                    [1190,1960,465,110],
                    [1470,950,660,120],
                    [275,1960,445,110],
                    [2665,1350,255,120],
                    [83,2264,450,98],
                    [870,950,590,120],
                    [1800,1350,845,120]])    
    
    
    #Dictionary containing the fields index row belonging to each policy field.
    fields_dict = {"body": 0,
            "chassis_serial": 1,
            "coverage": 2, #Per victim
            "coverage_limits": 3, #Limits given as number of daily minimum legal wages.
            "cylinder_cap": 4,
            "expiration_date": 5, #All policies expire at 23:59 of the given day.
            "fosyga_fee": 6,
            "inception_date": 7,
            "insurer": 8,
            "insurer_code": 9,
            "issue_date": 10, #All policies expire at 00:00 of the given day.
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
    
    for image in images_from_path:
      image.close()
