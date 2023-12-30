import cv2
import numpy as np
import face_recognition


img_rdj = face_recognition.load_image_file("image_attendance/ironman.jpg")
img_rdj = cv2.cvtColor(img_rdj, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file("image_basics/bgtest.jfif")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img_rdj)[0]
encodeRdj = face_recognition.face_encodings(img_rdj)[0]
cv2.rectangle(img_rdj, (faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]), (255, 0, 255), 2)

faceLoctest = face_recognition.face_locations(img_test)[0]
encodeRdjtest = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (faceLoctest[3],faceLoctest[0]), (faceLoctest[1],faceLoctest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeRdj],encodeRdjtest)
faceDis = face_recognition.face_distance([encodeRdj],encodeRdjtest)
#print(results)
#print(faceDis)
cv2.putText(img_test,f'{results[0]} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2)

#cv2.imshow("rdj", img_rdj)
cv2.imshow("rdj_test", img_test)
cv2.waitKey(0)
