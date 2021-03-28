import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
 
# list of users 
users = {"MANUEL":"215476966", "ELON":"215476969"}
# capture of userData
nameUser = input(str("Ingrese su nombre: "))
passWord = input(str("Contraseña: "))
continueFlag = True

path = './images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 


encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
counter = 0
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            if nameUser.upper() in name or nameUser.upper() == name:
                continueFlag = False

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow('webCam', img)
    if (counter > 200 and continueFlag):
        cap.release()
        cv2.destroyAllWindows()
        break
    elif not continueFlag:
        cap.release()
        cv2.destroyAllWindows()
        break
    counter += 1
    cv2.waitKey(1)

try:
    if (not continueFlag and passWord == users[nameUser.upper()]):
        print("Acceso concedido")
    else:
        print("Acceso denegado usuario o contraseña incorrecta")
except:
    print("Usuario incorrecto o inexistente")

