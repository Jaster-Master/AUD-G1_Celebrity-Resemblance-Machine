import os

import cv2
import time
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from icrawler.builtin import GoogleImageCrawler
from os.path import exists
import PIL

# https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/

# 0 is default webcam. Change to switch between multiple cameras or IP address.

projectDir = os.path.abspath(os.curdir)

imgDir = projectDir + '\\img'

vc = cv2.VideoCapture(0)

start_time = time.time()
display_time = 2
fc = 0
FPS = 0

while True:

    _, frame = vc.read()

    # imshow converts BGR to RGB while saving or displaying.
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF

    # press space to start -> 32 = Space
    if key == ord(chr(32)):

        detector = MTCNN()
        target_size = (224, 224)
        border_rel = 0

        detections = detector.detect_faces(frame)
        if len(detections) == 0:
            continue

        x1, y1, width, height = detections[0]['box']
        dw = round(width * border_rel)
        dh = round(height * border_rel)
        x2, y2 = x1 + width + dw, y1 + height + dh
        face = frame[y1:y2, x1:x2]

        cv2.imshow('MyFace', face)

        face = PIL.Image.fromarray(face)
        face = face.resize((224, 224))
        face = np.asarray(face)

        face_pp = face.astype('float32')
        face_pp = np.expand_dims(face_pp, axis=0)

        face_pp = preprocess_input(face_pp, version=2)

        model = VGGFace(model='resnet50')

        prediction = model.predict(face_pp)

        results = decode_predictions(prediction)

        print('Result:', results[0][0])

        promi_name = results[0][0][0].split('\'')[1].strip()

        print(promi_name)

        promi_file_name = imgDir + '\\' + promi_name + '.jpg'

        promi_file_exists = exists(promi_file_name)

        if promi_file_exists:
            celebrityImage = cv2.imread(promi_file_name)

            cv2.imshow('Celebrity', celebrityImage)
        else:

            google_Crawler = GoogleImageCrawler(storage={'root_dir': imgDir})

            google_Crawler.crawl(keyword=promi_name, max_num=1)

            os.rename(imgDir + '\\000001.jpg', promi_file_name)

            celebrityImage = cv2.imread(promi_file_name)

            celebrityImage = cv2.resize(celebrityImage, (224, 224), celebrityImage)

            cv2.imshow('Celebrity', celebrityImage)

            key = cv2.waitKey(1) & 0xFF
    else:
        # 27 = Escape
        if key == ord(chr(27)):
            break

cv2.destroyAllWindows()
