# Highway-Traffic-Analysis-for-Vehicle-Detection-and-Tracking
This project aims to develop a  traffic monitoring system that detects and tracks vehicles using deep learning techniques
Highway Traffic Analysis for Vehicle Detection and Tracking
📌 Overview
This project implements a traffic monitoring system capable of detecting and tracking vehicles using deep learning and computer vision techniques. It focuses on real-time vehicle counting and traffic flow analysis by leveraging the YOLOv8 object detection model and SORT tracking algorithm.

 Objectives
Detect vehicles such as cars, trucks, buses, and motorbikes using deep learning.

Track individual vehicles across video frames using a robust tracking algorithm (SORT).

Count vehicles accurately by detecting when they cross a predefined virtual line.

Provide basic traffic flow analysis using frame-wise object tracking.

 Methodology
1. Object Detection
Model: YOLOv8 (You Only Look Once)

Detection Pipeline:

A confidence threshold of 0.3 filters out low-confidence detections.

The model outputs bounding boxes, which are filtered by class labels (vehicle types).

A mask is applied to limit detection to the Region of Interest (ROI).

2. Object Tracking
Tracking Algorithm: SORT (Simple Online and Realtime Tracking)

Tracks detected vehicles frame-by-frame and assigns unique identifiers to avoid double-counting.

3. Vehicle Counting
A virtual line is defined across the video feed.

Vehicles crossing this line are counted once, based on their track ID.

 Implementation Details
 Programming Language
Python

 Libraries Used
OpenCV

cvzone

Ultralytics YOLO

filterpy

scikit-image

numpy
