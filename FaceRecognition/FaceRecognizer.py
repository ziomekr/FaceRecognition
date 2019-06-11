import numpy as np
import cv2
from PIL import Image
import os
import time

class FaceRecognizer:
    def __init__(self):
        #initialze models
        self.face_detector = cv2.dnn.readNetFromCaffe('model/deploy.prototxt', 'model/weights.caffemodel')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = []
    def find_faces(self, img):
        #find face coordinates in image
        (img_height, img_width) = img.shape[:2]
        if img_height < 300 or img_width < 300:
            return
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        try:
            detections = self.face_detector.forward()
        except:
            return
        face_boxes = []

        for i in range(0, detections.shape[2]):
            face_box = detections[0, 0, i, 3:7] * np.array([img_width, img_height, img_width, img_height])
            (startX, startY, endX, endY) = face_box.astype("int")             
            confidence = detections[0, 0, i, 2]
            if (confidence > 0.95):
                face_boxes.append((startX, startY, endX, endY))
                
        return face_boxes

    def get_training_images(self, path):
        #loading training data for training
        image_paths = [os.path.join(path,f) for f in os.listdir(path)]     
        face_samples=[]
        ids = []
        for image_path in image_paths:
            try:
                PIL_img = Image.open(image_path)
            except:
                continue
            img_numpy = np.array(PIL_img,'uint8')
            id = len(self.names)
            faces = self.find_faces(img_numpy)
            if faces is None:
                continue
            for (startX, startY, endX, endY) in faces:              
                face_samples.append(cv2.cvtColor(img_numpy[startY:endY,startX:endX], cv2.COLOR_BGR2GRAY))
                ids.append(id)
   
        return face_samples,ids

    def train_recognizer(self):
        #training recognizer with loaded images
        dirs = [os.path.join('data',f) for f in os.listdir('data')]
        for dir in dirs:
            faces, ids = self.get_training_images(dir)
            self.names.append(dir.split(sep='\\')[1])
            faces = [face for face in faces if face is not None]
            ids = ids[0:len(faces)]
            self.face_recognizer.update(faces, np.array(ids))
        self.face_recognizer.write('model/recognizer.yml')
    
    def capture_dataset_from_cam(self, username):
        #caputring images from webcam
        cam = cv2.VideoCapture(0)
        try:
            os.mkdir("data/"+str(username))
        except:
            print("User already exists, appending files to existing folder")
        count = 0
        while(count < 100):
            ret, img = cam.read()
            cv2.imwrite("data/"+str(username)+'/'+str(count)+'.jpg', img)
            cv2.imshow("Captured Image" + str(count), img)
            cv2.waitKey(100)
            count += 1
        cam.release()
        cv2.destroyAllWindows()

    def live_prediction(self):
        #finding and marking recognized faces from webcam view
        font = cv2.FONT_HERSHEY_SIMPLEX
        cam = cv2.VideoCapture(0)
        cam.set(3, 1280)
        cam.set(4, 1024)
        while(True):
            ret, img = cam.read()
            faces = self.find_faces(img)
            if faces is None:
                continue
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            for(startX, startY, endX, endY) in faces:
                cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
                try:
                    id, loss = self.face_recognizer.predict(gray[startY:endY,startX:endX])
                except:
                    continue
                if (loss < 50):
                    id = self.names[id]
                else:
                    id = "Unknown"        
                cv2.putText(img, str(id), (startX+5,startY-5), font, 1, (255,255,255), 2)
        
            cv2.imshow('img',img)
            key = cv2.waitKey(10) & 0xff
            if key == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
