import os

import fontTools.fontBuilder
from ultralytics import YOLO
import cv2
import cvzone
import math
#
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
cap = cv2.VideoCapture("../Videos/cars2.mp4")

model = YOLO("../YOLO-weights/yolov5nu.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    personCounter = 0
    motorbikeCounter = 0
    carCounter = 0
    largeVehiclesCounter = 0
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            print(box)
            # Class Name
            cls = int(box.cls[0])
            # Bounding box
            x, y, w, h = box.xywh[0]
            bbox = int(x - w / 2), int(y - h / 2), int(w), int(h)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # if cls == 0:
            #     cvzone.cornerRect(img, bbox, l=9)
            #     personCounter = personCounter+1
            #     # cvzone.putTextRect(img, f'{classNames[cls]}: {conf}', (max(0, int(x - w / 2)), max(35, int(y - h / 2 - 20))),
            #     #           scale=1.5, thickness=2, offset=3)
            #     cvzone.putTextRect(img, f'people: {personCounter}', (0, 50), scale=4, thickness=4, offset=3, colorR=(100,100,0))

            if cls == 3:
                cvzone.cornerRect(img, bbox, l=9)
                motorbikeCounter = motorbikeCounter + 1
                # cvzone.putTextRect(img, f'{classNames[cls]}: {conf}', (max(0, int(x - w / 2)), max(35, int(y - h / 2 - 20))),
                #           scale=1.5, thickness=2, offset=3)
                cvzone.putTextRect(img, f'{classNames[cls]}: {motorbikeCounter}', (0, 50), scale=4, thickness=4, offset=3, colorR=(100, 100, 0))

            if cls == 2:
                cvzone.cornerRect(img, bbox, l=9)
                carCounter = carCounter+1
                # cvzone.putTextRect(img, f'{classNames[cls]}: {conf}', (max(0, int(x - w / 2)), max(35, int(y - h / 2 - 20))),
                #           scale=1.5, thickness=2, offset=3)
                cvzone.putTextRect(img, f'{classNames[cls]}: {carCounter}', (0, 100), scale=4, thickness=4, offset=3, colorR=(100,100,0))

            if cls in [5, 7]:
                cvzone.cornerRect(img, bbox, l=9)
                largeVehiclesCounter = largeVehiclesCounter+1
                # cvzone.putTextRect(img, f'{classNames[cls]}: {conf}', (max(0, int(x - w / 2)), max(35, int(y - h / 2 - 20))),
                #           scale=1.5, thickness=2, offset=3)
                cvzone.putTextRect(img, f'large vehicles: {largeVehiclesCounter}', (0, 150), scale=4, thickness=4, offset=3, colorR=(100,100,0))
    cv2.imshow("Image", img)
    cv2.waitKey(1)