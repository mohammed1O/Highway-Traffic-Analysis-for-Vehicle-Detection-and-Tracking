#from tkinter import Image

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from  sort import*

#calculate Intersection over Union (IoU) between predicted and ground truth boxes
"""
def calculate_iou(box1, box2):
    """
    #box1: [x1, y1, x2, y2] For the expected box
    #box2: [x1, y1, x2, y2]  (Ground Truth)

"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

"""

# Open video stream
cap = cv2.VideoCapture(r"M:\Adobe\Project detection\cars.mp4")
# Load YOLO model
model = YOLO("M:/Adobe/Project detection/yolov8l.pt")


# Define class names for detection
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
# Initialize an empty list for counting
totalCount = []
# Load mask image
mask = cv2.imread("M:/Adobe/Project detection/mask1.png")


#tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [399 ,297,673,297]

while True:
    # Capture frame from video
    success, img = cap.read()
    # Apply mask to image
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)

    # Initialize an empty array for detected objects

    detections = np.empty((0, 5))
    # Loop through the results from the YOLO model
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Only process vehicles (car, truck, bus, motorbike)
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus"\
                    or currentClass == "motorbike" and conf > 0.3:
                #cvzone.putTextRect(img,f'{currentClass} {conf}', (max(0, x1), max(35,y1)),
                  #                 scale=0.6 , thickness=1 , offset=3)

               # cvzone.cornerRect(img,(x1 , y1, w, h), l=9)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))
    # Update tracker with new detections
    resultsTracker =tracker.update(detections)
    # Draw a line representing the crossing limits
    cv2.line(img,(limits[0], limits[1]), (limits[2], limits[3]), (0,0,255),4)
    # Loop through the tracked objects
    for results in resultsTracker:
        x1, y1, x2, y2, id =results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        print(results)
        # Draw bounding box and object ID
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img,(x1, y1, w, h), l=9 , rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f' {id}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # Calculate center of the bounding box
        cx,cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        # Check if the object crosses the limit line
        if limits[0]<cx< limits[2] and  limits[1]-20<cy<limits[1]+20:
            if totalCount.count(id)==0:
             totalCount.append(id)
             # Draw green line after crossing
             cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 4)
    # Display the total count of detected vehicles
    cvzone.putTextRect(img, f'count: {len (totalCount)}', (50, 50)),


    cv2.imshow("Image", img)
    #cv2.imshow("ImagRegion", imgRegion)
    # Wait for 1 ms before moving to the next fram
    cv2.waitKey(1)