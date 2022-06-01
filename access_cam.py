import math

import numpy as np
import cv2
import os
import time
import imutils
import dlib
from imutils.video import VideoStream, FPS
from centroid_tracker import CentroidTracker
from trackable_object import TrackableObject

# NOTE: Install cmake, miniconda and dlib:
#   conda install -c conda-forge dlib


class PedestrianDetector:
    # TODO: Have Centroid tracker predict along a centroid's path to reacquire lost targets

    NMS_THRESHOLD = 0.3
    MIN_CONFIDENCE = 0.3
    APPLY_AI_EVERY = 10
    FRAMES_BEFORE_DISAPPEAR = (APPLY_AI_EVERY*3)+1
    DISTANCE_BEFORE_NEW_OBJECT = (20*APPLY_AI_EVERY)

    RESIZE_WORKING_IMAGE = int(3840 / 2)
    USE_CUDA_GPU = False

    _coco_labels_path = "coco.names"
    _weights_path = "yolov4-tiny.weights"
    _config_path = "yolov4-tiny.cfg"
    # insert the HTTP(S)/RSTP feed from the camera
    # stream_url = "rtsp://admin:PW@@192.168.1.62/Streaming/Channels/102"
    # stream_url = "videos/annke2hd.20220529_220302_1.mp4"
    stream_url = "videos/annke2hd.20220529_235055_1.mp4"
    # stream_url = "videos/reolink5hd.20220529_201349_1.mp4"

    file_to_export_video_to = ""

    fps = None
    writer = None
    person_trackers = []
    trackable_objects = {}
    centroid_tracker = None
    person_rectangles = []
    fourcc = None
    status = None
    total_frames = 0

    total_up = 0
    total_down = 0

    def __init__(self):
        self.initialize()

    def initialize(self):
        # Setup the model and person recognizer neural net
        layer_name, model, person_layer_id = self.setup_model()

        # Open the video
        print("[INFO] starting video stream...")
        stream = cv2.VideoCapture(self.stream_url)

        # Create a writer object that could be used for saving
        self.writer = None
        self.person_trackers = []
        self.trackable_objects = {}
        self.person_rectangles = []
        self.centroid_tracker = CentroidTracker(maxDisappeared=self.FRAMES_BEFORE_DISAPPEAR,
                                                maxDistance=self.DISTANCE_BEFORE_NEW_OBJECT)
        self.W = None
        self.H = None
        self.total_up = 0
        self.total_down = 0
        self.status = "No one detected"
        self.total_frames = 0

        # Start tracking video speed
        self.fps = FPS().start()
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break

            # Make the image smaller to work with and convert to RGB color system
            frame = imutils.resize(frame, width=self.RESIZE_WORKING_IMAGE)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]
                self.centroid_tracker.maxDistance = self.W / 8
                print("Resized to: {} {}".format(self.H, self.W))

            # Create the writer file and object if needed
            if not self.file_to_export_video_to == "" and self.writer is None:
                self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.writer = cv2.VideoWriter(self.file_to_export_video_to, self.fourcc, 30,
                                              (frame.shape[1], frame.shape[0]), True)

            self.person_rectangles = []

            # Run the computationally expensive AI every few frames
            if self.total_frames % self.APPLY_AI_EVERY == 0:
                self.status = "Detecting"
                self.person_trackers = []

                # Use the NN to find boxes that meet a minimum confidence
                pedestrian_boxes = self.pedestrian_detection(frame, model, layer_name, person_ids=person_layer_id)

                for _box in pedestrian_boxes:
                    startX, startY, endX, endY = int(_box[1][0]), int(_box[1][1]), int(_box[1][2]), int(_box[1][3])

                    # Trim the box that the learning algorithm will focus on
                    startX, startY, endX, endY = self.shorten_box(startX, startY, endX, endY)

                    tracker = dlib.correlation_tracker()
                    _rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, _rect)

                    self.person_trackers.append(tracker)
                    self.person_rectangles.append((startX, startY, endX, endY))

            else:  # Run on the "non AI processing" frames
                # Predict where all previous trackers say people will be
                self.status = "Tracking"
                for _track in self.person_trackers:
                    _track.update(rgb)  # TODO: Add in dead reckoning prediction here?
                    dlib_image_guess = _track.get_position()

                    startX, startY = int(dlib_image_guess.left()), int(dlib_image_guess.top())
                    endX, endY = int(dlib_image_guess.right()), int(dlib_image_guess.bottom())

                    self.person_rectangles.append((startX, startY, endX, endY))

            # Get new guesses of where people will be
            centroid_loc_tracker_guesses = self.centroid_tracker.update(self.person_rectangles)

            for (objectID, centroid) in centroid_loc_tracker_guesses.items():
                # check to see if a trackable object exists for the current object ID
                _trackable_obj = self.trackable_objects.get(objectID, None)
                if _trackable_obj is None:
                    _label = "Person {}".format(objectID+1)
                    _trackable_obj = TrackableObject(objectID, centroid, label=_label, w=self.W, h=self.H)

                # otherwise, there is a trackable object that we can utilize it to determine direction
                else:
                    # the difference between the y-coordinate of the *current* centroid and the mean of *previous*
                    # centroids tells us which direction the object is moving (negative for 'up' and pos for 'down')
                    y = [c[1] for c in _trackable_obj.centroids]
                    direction = centroid[1] - np.mean(y)  # TODO: Use for target prediction
                    _trackable_obj.add_centroid(centroid)
                    _trackable_obj.last_rect = self.closest_rect(centroid)

                    # check to see if the object has been counted or not
                    if not _trackable_obj.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0 and centroid[1] < self.H // 2:
                            self.total_down += 1
                            _trackable_obj.counted = True
                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction > 0 and centroid[1] > self.H // 2:
                            self.total_up += 1
                            _trackable_obj.counted = True

                # (re)store the trackable object in our dictionary
                self.trackable_objects[objectID] = _trackable_obj

                mid_x = centroid[0]
                mid_y = centroid[1]

                lp_next = [mid_x, mid_y]
                for count, lp in enumerate(_trackable_obj.centroids):
                    if count < len(_trackable_obj.centroids)-1:
                        lp_next = _trackable_obj.centroids[count+1]
                        cv2.line(frame, (int(lp[0]), int(lp[1])), (int(lp_next[0]), int(lp_next[1])),
                                 _trackable_obj.color)

                cv2.circle(frame, (lp_next[0], lp_next[1]), 4, (0, 255, 0), -1)

            # Show all objects that dlib is tracking
            for _obj_key in self.trackable_objects.keys():
                _trackable_obj = self.trackable_objects[_obj_key]

                _confidence = "0"
                _color = _trackable_obj.color
                _text = "{} [{}%]".format(_trackable_obj.label, _confidence)
                _r = _trackable_obj.last_rect
                if _r:
                    if not type(_r) == tuple:  # If it's a dlib rect, pull out the rectangular values
                        _r = (_r.left(), _r.top(), _r.right(), _r.bottom())
                    cv2.putText(frame, _text, (_r[0] - int(self.H/70), _r[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _color, 2)
                    cv2.rectangle(frame, (_r[0], _r[1]), (_r[2], _r[3]), _color, 2)
                else:
                    cv2.putText(frame, _text, (mid_x - int(self.H/70), mid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                _color, 2)

            # loop over the info tuples and draw them on our frame
            self.hud_info(True, frame)

            # Check to see if we should write the frame to disk
            if self.writer is not None:
                self.writer.write(frame)

            # Show the frame to screen
            cv2.imshow("Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # User hit q
                break

            self.total_frames += 1
            self.fps.update()

        # Stop the timer and display FPS information
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        # loop over the info tuples and draw them on our frame
        self.hud_info(False)

        # Print out the points of the first path
        # _trackable_obj = self.trackable_objects.get(1, None)
        # x_list = []
        # y_list = []
        # for count, lp in enumerate(_trackable_obj.centroids):
        #     x_list.append(int(lp[0]))
        #     y_list.append(int(lp[1]))
        # import json
        # print("x_list = {}".format(json.dumps(x_list)))
        # print("y_list = {}".format(json.dumps(y_list)))

        # Check to see if we need to release the video writer pointer
        if self.writer is not None:
            self.writer.release()

        # close the connection and close all windows
        stream.release()
        cv2.destroyAllWindows()

    @staticmethod
    def shorten_box(_l, _t, _r, _b):
        # return the top half of a box and middle 80%
        _h = _b - _t
        _w = _r - _l

        _b = int(_b - (_h*.5))
        _l = int(_l + (_w*.2))
        _r = int(_r - (_w*.2))

        return _l, _t, _r, _b

    def closest_rect(self, centroid):
        closest = None
        dist_closest = 1000000

        for r in self.person_rectangles:
            cX = int((r[0] + r[2]) / 2.0)
            cY = int((r[1] + r[3]) / 2.0)

            dist = math.dist(centroid, [cX, cY])
            if dist < dist_closest:
                closest = r
                dist_closest = dist

        return closest


    def hud_info(self, show_on_cv_frame, frame=None):
        _hud_info = [
            ("Going Up past center", self.total_up),
            ("Going Down past center", self.total_down),
            ("People Seen", len(self.trackable_objects)),
            ("Status", self.status),
        ]
        if show_on_cv_frame:
            for (i, (k, v)) in enumerate(_hud_info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            for (i, (k, v)) in enumerate(_hud_info):
                text = "[INFO] {}: {}".format(k, v)
                print(text)

    def setup_model(self):
        _coco_labels = open(self._coco_labels_path).read().strip().split("\n")
        model = cv2.dnn.readNetFromDarknet(self._config_path, self._weights_path)
        if self.USE_CUDA_GPU:
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        layer_name = model.getLayerNames()
        layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

        person_layer_id = _coco_labels.index("person")

        return layer_name, model, person_layer_id

    def pedestrian_detection(self, image, model, layer_name, person_ids=0):
        results = []

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        model.setInput(blob)
        layer_outputs = model.forward(layer_name)

        boxes = []
        centroids = []
        confidences = []

        for output in layer_outputs:
            for detection in output:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == person_ids and confidence > self.MIN_CONFIDENCE:

                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        likely_boxes = cv2.dnn.NMSBoxes(boxes, confidences, self.MIN_CONFIDENCE, self.NMS_THRESHOLD)
        # ensure at least one detection exists
        if len(likely_boxes) > 0:
            # loop over the indexes we are keeping
            for i in likely_boxes.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # update our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                res = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(res)
        # return the list of results
        return results


PD = PedestrianDetector()
