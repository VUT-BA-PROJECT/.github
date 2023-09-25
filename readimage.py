import cv2
import numpy as np

# Load the pre-trained license plate detection model
numberPlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml') 
plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Load the image
img = cv2.imread('Data/img1.jpg')

# Convert the image to grayscale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect license plates in the image
plates = plat_detector.detectMultiScale(img,scaleFactor=1.2,
    minNeighbors = 5, minSize=(25,25))   

# Loop through the detected plates and draw rectangles around them
for (x,y,w,h) in plates:
    cv2.putText(img,text='License Plate',org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=1,fontScale=0.6)
    img[y:y+h,x:x+w] = cv2.blur(img[y:y+h,x:x+w],ksize=(10,10))
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
  
# Display the image with detected plates
cv2.imshow('plates',img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
