import argparse
import imutils
from imutils.video import VideoStream
import time
import cv2
import numpy as np
import os
from keras.models import load_model
from config import *
from utils import get_yolo_boxes

def detect_face(frame): 
    h, w = frame.shape[:2]
    boxesObj = get_yolo_boxes(model, [frame], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]
    boxes = []
    for box in boxesObj:
        label_str = ''
        label = -1

        if box.c > obj_thresh:
            if label_str != '': label_str += ', '
            label_str += (str(round(box.c * 100, 2)) + '%')
            label = 0

        if label >= 0:

            boxes.append((box.xmin, box.ymin, box.xmax-box.xmin, box.ymax-box.ymin))

    return(boxes)

def draw_boxes(boxes, frame, color):
    ymax,xmax,_ = np.shape(frame)
    for box in boxes:
        (x, y, w, h) = [int(v)*enlarge for v in box]
        x = max(x,0)
        y = max(y,0)
        x_= min(xmax, x+w)
        y_= min(ymax,y+h)
        cv2.rectangle(frame, (x, y), (x_, y_), color, 2)
        # face_image = frame[y:y_, x:x_]
        # face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
        # frame[y:y_, x:x_] = face_image

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', default="../face_detection_in_realtime/weights/shufflenetv2.h5",
                help='path to pre-trained weights.')
    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="mosse",
        help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    global model 
    model = load_model(args['model'])

    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # initialize OpenCV's special multi-object tracker
    trackers = cv2.MultiTracker_create()

    videoWriter = cv2.VideoWriter('face_blur2.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))

    # if a video path was not supplied, grab the reference to the web cam
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])


    global enlarge
    enlarge = 4 
    i = 0 
    # loop over frames from the video stream
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        
        i += 1
        if i < 400:
            continue

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1/enlarge, fy=1/enlarge)


        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked
        
        (success, boxes_track) = trackers.update(small_frame)
        boxes_rec = detect_face(small_frame)
        
        print(boxes_track,success)

        if not success or boxes_track == () or len(boxes_rec)>len(boxes_track):
            boxes_rec = detect_face(small_frame)
            
            # initialize OpenCV's special multi-object tracker
            trackers = cv2.MultiTracker_create()
            for box in boxes_rec:
                tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                trackers.add(tracker, small_frame, box)

            boxes_track = []


        if success:
            pass
        # loop over the bounding boxes and draw then on the frame
        draw_boxes(boxes_track, frame, (0,255,0))
        draw_boxes(boxes_rec, frame, (255,0,0))
        
       
        # show the output frame
        cv2.imshow("Frame", frame)
        videoWriter.write(frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    if not args.get("video", False):
        vs.stop()
    else:
        vs.release()

    # close all windows
    cv2.destroyAllWindows()
