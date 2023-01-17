import pytesseract
from PIL import Image
import cv2
from gtts import gTTS
import playsound 


camera = cv2.VideoCapture(0)

while True:
    ret,image=camera.read()
    cv2.imshow("book",image)
    #cv2.imwrite('book.jpg',image)
    cv2.waitKey(5000)
    break

camera.release()
cv2.destroyAllWindows()   


def tessarect():
    
    pytesseract.pytesseract.tesseract_cmd= r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    img=Image.open(r"E:\Robotics _Competetion\book.jpg")
    
    text= pytesseract.image_to_string(img,lang='ben')
    print(text)
    
    tst= gTTS(text,lang='bn')
    tst.save("read.mp3")
    playsound.playsound('read.mp3')

tessarect()    

