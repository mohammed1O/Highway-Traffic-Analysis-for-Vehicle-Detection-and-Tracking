#Project Name: Vehicle Detection and Tracking System

Description
This project uses the YOLO (You Only Look Once) object detection model to detect and track vehicles  in  video . The system also counts vehicles as they cross a defined line within the video frame. It uses OpenCV for video processing and cvzone for visual enhancements, and it incorporates the SORT tracker for object tracking.




  Libraries  -------------
YOLOv8: For vehicle detection.

OpenCV: For video processing and image manipulation.


cvzone: For drawing bounding boxes and text.

scikit-image>> pip install scikit-image 

SORT: For object tracking, taken from this repository "https://github.com/abewley/sort".  
filterpy >> pip install filterpy

NumPy: For numerical operations.

math: For calculate mathematical values

filterpy:()

-----------------------------------------

File Structure

/Vehicle-Detection-Tracking/
    ├── main.py           # Main script for vehicle detection and tracking
    ├── mask1.png                # Mask image for region focus
    ├── cars.mp4                 # Input video file
    ├── yolov8l.pt               # YOLOv8 model weights
    └── sort.py                  # object tracking
    └── yolov8l.pt                # YOLO model
    
--------------------------------------------
How to Run
Install dependencies as . (all  Libraries)
{
pip install numpy
pip install ultralytics
pip install cvzone
pip install scikit-image
pip install filterpy

}

Run the system:

>> python main.py

   A window will open showing the video feed with vehicle detection and tracking.

