import cv2
import time
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import PIL

# 0 is default webcam. Change to switch between multiple cameras or IP address.

vc = cv2.VideoCapture(0)

start_time = time.time()
# FPS update time in seconds
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

        # Show Promi face

        key = cv2.waitKey(1) & 0xFF
    else:
        # 27 = Escape
        if key == ord(chr(27)):
            break

cv2.destroyAllWindows()
# https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/

