# Real-time Face Tracker
This repository is the implementation of face detection and pattern track to realize realtime face tracking with high FPS. I took [opconty's work](https://github.com/opconty/face_detection_in_realtime) as a reference for the main part of face detection.

## Requirements
- tensorflow
- keras
- cv2

## Usage

Default using webcam's live stream.

``
python face_track.py
``

 Or you can use below command to process video.

``
python face_track.py -v <path-to-video>
``

## Demo Result

