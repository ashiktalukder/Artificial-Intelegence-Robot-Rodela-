import speech_recognition as sr
import pyttsx3
import sounddevice
import pywhatkit
from gtts import gTTS
import playsound
import serial


import pygame 
import time
#Joks#####
import requests
from bs4 import BeautifulSoup
import lxml
from gtts import gTTS
import playsound 
#jok###

###openCv####
import cv2 as cv
import numpy as np
import os
import pytesseract
####end####

listener = sr.Recognizer()
engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)

def talkeng(text):
    engine.say(text)
    #engine.say(command)
    engine.runAndWait()
def talk(text):
    tst= gTTS(text,lang='bn')
    tst.save("say.mp3")
    playsound.playsound('say.mp3')

talk('আমি রোদেলা... আমি কিভাবে আপনাকে সাহায্য করতে পারি ?')

def take_command():
    try:
        with sr.Microphone() as source:
        
            listener.adjust_for_ambient_noise(source,duration=1)   
        
            print('বলুন')
            voice = listener.listen(source)
            command = listener.recognize_google(voice,language='bn-BD')
            #talk("আচ্ছা ঠিক আছে")
            print(command)
            command = command.lower()
       
            if 'রোদেলা' in command:
                command = command.replace('রোদেলা' , '')
                #talk(command)
                print(command)
                
    except:
        print('আবার বলুন')
        run_rodela()
        pass
    return command

##################jokes Point########
def joks():
    
    webpage=requests.get('http://uits.ww5.co/UITS/?page_id=5').text
    soup=BeautifulSoup(webpage,'lxml')
    #tag=soup.find('div' , class='entry-content').text
    for i in soup.find_all('p'):
        read= i.text.strip()
        start="পচা আর তার প্রেমিকা একদিন এক পার্কে বসেছিল।"
        end ="ইয়ে মানে প্রস্রাব করে দিই !"
        jokes= read[read.find(start):read.find(end)+ len(end)]
        #print(jokes)

    tst= gTTS(jokes,lang='bn')
    tst.save("joks.mp3")
    playsound.playsound('joks.mp3')
    

##################end Jokes############

######  --5 -Document like wikipedia##########
def seikh_hasina():
    with open('document\seikhhasina.txt', 'r', encoding='utf-8') as file:
        lines = file.read()
        #print(lines.strip())
        txt= gTTS(lines,lang='bn')
        txt.save("hasina.mp3")        
        pygame.init()
        pygame.mixer.music.load('hasina.mp3')
        pygame.mixer.music.play()
           
def computer():
    with open('document\computer.txt', 'r', encoding='utf-8') as file:
        lines = file.read()
        #print(lines.strip())
        txt= gTTS(lines,lang='bn')
        txt.save("com.mp3")        
        pygame.init()
        pygame.mixer.music.load('com.mp3')
        pygame.mixer.music.play() 
#####endDocument



### Face Data Train ################################

def train_face():
    people =[]
    for i in os.listdir('images'):
        people.append(i)
        #print(people)
    DIR = 'images'
    #print(os.listdir)
    haar_cascade = cv.CascadeClassifier('facedetect.xml')

    features = []
    labels = []

    def create_train():
        for person in people:
            path = os.path.join(DIR, person)
            label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

        create_train()
        print('Training done ---------------')

    #print(f'length of feature = {len(features)}')
    #print(f'length of label = {len(labels)}')

    features = np.array(features, dtype='object')
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    # Train the Recognizer on the features list and the labels list
    face_recognizer.train(features,labels)

    face_recognizer.save('face_trained.yml')
    

### End Face Data Train ############################
##### Face Datection##########
def face_detect():
    people =[]
    for i in os.listdir('images'):
        people.append(i)
    #print(people)
    DIR = 'images'
    
    #haar_cascade = cv.CascadeClassifier('cascadeashik.xml')
    haar_cascade = cv.CascadeClassifier('facedetect.xml')
    
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_trained.yml')
    cap = cv.VideoCapture(0)

    while True:
        # Capture a single frame
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)      

    # Detect the face in the image
        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        #print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(frame, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

        #cv.imshow('Detected Face', frame)
        #print(f'Label = {people[label]} with a confidence of {confidence}')
        if(confidence>40):
            Label = {people[label]}
            talkeng('hi')
            talkeng(Label)
            tst= gTTS(text=('কেমন আছেন?'),lang='bn')
            tst.save("hru.mp3")
            playsound.playsound('hru.mp3')
            ##conect the port than remove command it's work
            #ser = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)
            ###ser.write(b'Hand-UP\n')               
            ##ser.close() 
            break
        elif(confidence<40):
            tst= gTTS(text=('আমি চিনতে পারছি না'),lang='bn')
            tst.save("cinsina.mp3")
            playsound.playsound('cinsina.mp3')
            print("ছবি তুলুন দয়া করে")
            capture_data()               
            break
        #if cv.waitKey(1) & 0xFF == ord('q'):
           # break
        
    cap.release()
    cv.destroyAllWindows()

##### End Face Datection##########

####### Creat New Face Data######
def capture_data():
    
    listener = sr.Recognizer()
    with sr.Microphone() as source:
        
            listener.adjust_for_ambient_noise(source,duration=1)   
        
            print('লোকটির নাম কি')
            tst= gTTS(text=('লোকটির নাম কি'),lang='bn')
            tst.save("askname.mp3")
            playsound.playsound('askname.mp3')
            voice = listener.listen(source)
            command = listener.recognize_google(voice,language='en')            
            
            command = command.lower()
            print(command)    
    ## please ;;; it's My won Make data set if better result use casecade dataset
    face_classifier = cv.CascadeClassifier('cascadeashik.xml')

    def face_extractor(img):

        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]

            return cropped_face
        
    cap = cv.VideoCapture(0)
    count = 0
    os.mkdir("images/" + command)
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv.resize(face_extractor(frame),(300,300))
                      
            file_name_path =  str(count)
            cv.imwrite(f"images/{command}/{file_name_path}.jpg", face)
            
            cv.putText(face,str(count),(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv.imshow('Face Cropper',face)
            time.sleep(1)
        else:
            print("Face not found")
            pass
        if count==10:
            break

    cap.release()
    cv.destroyAllWindows()
    print('capture face data complete')
    train_face()

###endd Capture Image######

## Read Book#####
def read_book():
    
    print("দয়া করে বইটি কাসে নিয়া ধরুন")
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    cap = cv.VideoCapture(0) 
    while True:
        # Capture frame
        ret, frame = cap.read()    
        cv.imshow('Frame', frame)        
        text = pytesseract.image_to_string(frame,lang='ben')
        print(text)        
        cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
                
        tts = gTTS(text, lang='bn')
        # Save the audio file
        tts.save('bokread.mp3')
        #playsound.playsound('bookread.mp3')
        pygame.init()
        pygame.mixer.music.load('bokread.mp3')
        pygame.mixer.music.play()
        
        command=take_command()
        if 'থেমে' or 'থামো' in command:
            pygame.mixer.music.stop()
            break

    cap.release()   
    cv.destroyAllWindows()

## end Read Book####


def run_rodela():
    command = take_command()
    if 'বাজাও' in command:
        song = command.replace('বাজাও','')
        pywhatkit.playonyt(song)        
        print(song)
        #run_robo()
    
        
    elif 'শেখ হাসিনা সম্পর্কে' in command:
        seikh_hasina()
        time.sleep(20)
        pygame.mixer.music.stop()
        
        
    elif 'শেখ হাসিনা বিস্তারিত' in command:
            pygame.init()
            pygame.mixer.music.load('hasina.mp3')
            pygame.mixer.music.play()
               
    elif 'কম্পিউটার সম্পর্কে' in command:
        computer()
        time.sleep(20)
        pygame.mixer.music.stop()
        
        
    elif 'কম্পিউটার বিস্তারিত' in command:
            computer()
            pygame.init()
            pygame.mixer.music.load('com.mp3')
            pygame.mixer.music.play()
            
    elif 'কৌতুক' in command:
        joks()
        
    elif 'কেমন আছো' in command:
        tst= gTTS(text=('আমি ভালো আছি....আপনি কেমন আছেন?'),lang='bn')
        tst.save("fine.mp3")
        playsound.playsound('fine.mp3')

    elif 'ভালো আছি' in command:
        tst= gTTS(text=('ধন্যবাদ'),lang='bn')
        tst.save("thanks.mp3")
        playsound.playsound('thanks.mp3')
    
    elif 'লোকটিকে চেনো' in command:
        face_detect()
        
    elif 'বইটি পড়ে শোনাও' in command:
        read_book()    
        
    elif 'শুনতে চাচ্ছি না' in command:
            pygame.mixer.music.stop()    
    else:
        
        tst= gTTS(text=('আমি বুঝতে পারিনি....আবার বলুন'),lang='bn')
        tst.save("folt.mp3")
        playsound.playsound('folt.mp3')
        run_rodela()
        #playsound.playsound('folt.mp3')
    
    
while True: 
    run_rodela()


      