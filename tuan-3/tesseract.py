import cv2 
import pytesseract

img = cv2.imread('image_test.jpg')
print(pytesseract.image_to_string(img))
