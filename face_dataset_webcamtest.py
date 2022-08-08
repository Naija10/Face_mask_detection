from tensorflow import keras
import numpy as np
import cv2



dataset=np.load("DATASETS/TRAIN.npy",allow_pickle=True)
classes=["mask","nomask"]

video=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
model=keras.models.load_model("Models/best_of_facemask.hdf5")
while True:
     ret,frame=video.read()
     faces=face_cascade.detectMultiScale(frame)
     for (x,y,w,h) in faces:
          roi=frame[y:y+h,x:x+w].copy()
          cv2.imwrite("webcamimages/face.jpg",roi)
          #test_image=image.load_img("Faceimages/webcamimages/face.jpg",target_size=(50,50,3))
          test_image=cv2.resize(roi,(50,50))
          test_image=np.array(test_image)
          test_image=np.expand_dims(test_image,axis=0)
          n_test_image=test_image/255
          prediction=model.predict_classes(n_test_image)[0][0]
          if prediction==0:
               cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
               cv2.putText(frame,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
          else:
               cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
               cv2.putText(frame,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
     cv2.imshow("result",frame)
     if cv2.waitKey(15)==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
