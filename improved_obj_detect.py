import cv2
import os
import pygame
import time
import numpy as np
import urllib.request
from gtts import gTTS
import requests

# For ESPCAM IP
url = 'http://192.168.254.63/cam-hi.jpg'

cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

whT = 320

# ML model
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = 'coco.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# text-to-speech function for announcements
def Text_to_speech(string):
    language = 'en'
    myobj = gTTS(text=string, lang=language, slow=False)
    myobj.save("speech.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("speech.mp3")
    pygame.mixer.music.play()
    time.sleep(8)
    pygame.mixer.music.unload()
    os.remove('speech.mp3')

# realtime distance fetched from the nodemcu
def fetch_distance():
    try:
        response = requests.get('http://192.168.254.156/distance')  # nodemcu ip address
        if response.status_code == 200:
            return response.text
        else:
            print("Failed to fetch distance data")
            return "0 cm"
    except Exception as e:
        print(f"Error fetching distance data: {e}")
        return "0 cm"

# object detections function
def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        text_speech = f'{classNames[classIds[i]].upper()}'
        distance = fetch_distance()  # Get the distance from NodeMCU
        message = f"Detected {text_speech} at {distance} away."
        Text_to_speech(message)

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)
    sucess, img = cap.read()
    blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    findObject(outputs, im)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    cv2.imshow('Object identification', im)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
