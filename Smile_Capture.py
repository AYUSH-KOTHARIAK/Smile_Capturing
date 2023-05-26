import numpy as np
import cv2
import matplotlib.pyplot as py
from time import sleep  
fd=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
Is=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
vid = cv2.VideoCapture(0)
notCaptured = True
seq=0
while notCaptured:
    flag,img = vid.read()
    if flag:
        ##Processing code
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
        faces = fd.detectMultiScale(img_gray,
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize = (50,50))   ## -- (imagename,scale factor,minNeighbors,minSize)
        np.random.seed(50)
        colour = np.random.randint(0,255,(len(faces), 3)).tolist()
        i=0

        for x,y,w,h in faces:
            face=img_gray[y:y+h,x:x+w].copy()

            smiles = Is.detectMultiScale(face,
                                         scaleFactor = 1.1,
                                         minNeighbors = 5,
                                         minSize =(50,50))
            if len(smiles) == 1:
                seq +=1
                if seq ==10 :  ## for 10 frames with smile then capture
                     cv2.imwrite('myselfie.png',img)    ## to capture
                     notCaptured = False  # capture and break
                     break 
            cv2.rectangle(
                img,pt1=(x,y),pt2=(x+w,y+h),color=colour[i],
                thickness=2
            )
            i +=1
        cv2.imshow('preview',img)    
        key=cv2.waitKey(1)
        if key == ord('q'):   ## key to exit the priview    ,, ord('') -- gives ASCII value of the keys or we can also replace it by its ASCII value
            break
    else:
        print('NO FRAME')
        break  ## if not reading any input from camera
    sleep(0.1)
cv2.destroyAllWindows() 
cv2.waitKey(1)
vid.release() ## to release camera access by code