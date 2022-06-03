import imutils
from imutils.video import VideoStream, FPS
import cv2
import numpy as np
import dlib
from matplotlib import colors

import math
import yaml
import json
import os
from argparse import ArgumentParser

from centroid_tracker import CentroidTracker
from trackable_object import TrackableObject


# INSTALL NOTE: Install all libraries.  If on mac, install cmake before dlib.  If on PC, install VS studio and C++
#   build tools and build one project using CMake.  Then pip install dlib


class PedestrianDetector:
    # TODO: Have Centroid tracker predict along a centroid's path to reacquire lost targets, then only feed that to dlib

    # Default variables, also can be set in config.yaml
    NMS_THRESHOLD = 0.3
    MIN_CONFIDENCE = 0.3
    APPLY_AI_EVERY = 10
    FRAMES_BEFORE_DISAPPEAR = (APPLY_AI_EVERY * 3) + 1
    DISTANCE_BEFORE_NEW_OBJECT = (20 * APPLY_AI_EVERY)
    RESIZE_WORKING_IMAGE = int(3840 / 2)
    USE_CUDA_GPU = False

    DETECTORS = [
        {"detector_title": "person", "box_width": 4, "line_width": 2,"color": (255, 0, 0), "reduce_box": True},
    ]

    _coco_labels_path = "coco.names"
    _weights_path = "yolov4-tiny.weights"
    _config_path = "yolov4-tiny.cfg"
    # insert the HTTP(S)/RSTP feed from the camera
    # stream_url = "rtsp://admin:PW@@192.168.1.62/Streaming/Channels/102"
    stream_url = None
    current_video_name = ""
    video_directory = "videos"

    fps = None
    writer = None
    person_trackers = []
    trackable_objects = {}
    centroid_tracker = None
    person_rectangles = []
    detector_types = []
    fourcc = None
    status = None
    total_frames = 0

    nn_layer_name = None
    model = None
    nn_person_layer_id = None

    total_up = 0
    total_down = 0

    viewer_show_lines = False
    viewer_show_boxes = True
    viewer_show_individual_colors = False

    def load_configuration(self):

        parser = ArgumentParser(description="Look for people and objects in video files or streams")
        parser.add_argument("-c", "--config", dest="config_filename", default="config.yaml",
                            help="config yaml file to read settings from", metavar="FILE")
        parser.add_argument("-s", "--stream", dest="stream_url", default=None,
                            help="stream url to use instead of local videos")
        parser.add_argument("-d", "--dir", dest="video_directory", default="videos",
                            help="local directory of videos to parse")

        args = parser.parse_args()
        self.stream_url = args.stream_url
        self.video_directory = args.video_directory

        if args.config_filename:
            if os.path.exists(args.config_filename):
                with open(args.config_filename, 'r') as conf_file:
                    d = yaml.safe_load(conf_file)
                    print("Loading YAML file {}".format(conf_file))

                    settings_new = d['tracker']['settings'] if 'tracker' in d and 'settings' in d['tracker'] else {}
                    detectors_new = d['tracker']['detectors'] if 'tracker' in d and 'detectors' in d['tracker'] else {}
                    if 'nms_threshold' in settings_new:
                        self.NMS_THRESHOLD = settings_new['nms_threshold']
                    if 'min_nn_confidence' in settings_new:
                        self.MIN_CONFIDENCE = settings_new['min_nn_confidence']
                    if 'apply_ai_every' in settings_new:
                        self.APPLY_AI_EVERY = settings_new['apply_ai_every']
                    if 'distance_before_new_object' in settings_new:
                        self.DISTANCE_BEFORE_NEW_OBJECT = settings_new['distance_before_new_object']
                    if 'video_directory' in settings_new:
                        self.video_directory = settings_new['video_directory']
                    if 'resize_scanning_video_to_width' in settings_new:
                        self.RESIZE_WORKING_IMAGE = settings_new['resize_scanning_video_to_width']
                    if 'use_cuda_on_gpu' in settings_new:
                        self.USE_CUDA_GPU = settings_new['use_cuda_on_gpu']
                    print("..settings parsed")

                    if len(detectors_new):
                        self.DETECTORS = []
                        for d_category in detectors_new:
                            d = detectors_new[d_category]
                            d_item = {'detector_title': d_category,
                                      'box_width': d['box_width'] if 'box_width' in d else 1,
                                      'line_width': d['line_width'] if 'line_width' in d else 1,
                                      'color': d['color'] if 'color' in d else 'red',
                                      'reduce_box': d['reduce_box'] if 'reduce_box' in d else False}
                            self.DETECTORS.append(d_item)
                    print("..{} detectors added".format(len(self.DETECTORS)))

        # Change color words to values
        for d in self.DETECTORS:
            if 'color' in d and type(d['color'] == str):
                col = colors.to_rgb(d['color'])
                d['color'] = [int(i*255) for i in col]

        print("Configuration set")

    def __init__(self):
        self.initialize()

    def initialize(self):
        # Load settings from command line and conf file
        self.load_configuration()

        # Setup the model and person recognizer neural net
        self.nn_layer_name, self.model = self.setup_model()

        if self.stream_url:
            print("[INFO] starting video stream...")
            user_let_finish = self.play_video(self.stream_url)
        else:
            # Loop through all videos in video directory
            file_name = None
            user_let_finish = True
            while user_let_finish:
                file_name = next_file_in_dir(current_file=file_name, directory=self.video_directory)
                user_let_finish = self.play_video("{}/{}".format(self.video_directory, file_name))

        # When done, have opencv close the video window
        cv2.destroyAllWindows()

    def play_video(self, stream_link, file_to_export_video_to=""):
        # Open the video
        stream = cv2.VideoCapture(stream_link)
        self.current_video_name = stream_link
        # Don't print stream title in case password is with URL
        print("[INFO] -- Loaded video: {}".format("Stream" if self.stream_url else stream_link))

        # Create a writer object that could be used for saving
        self.writer = None
        self.person_trackers = []
        self.trackable_objects = {}
        self.person_rectangles = []
        self.detector_types = []
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
        user_let_finish = True
        paused = False
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break

            # Make the image smaller to work with
            frame = imutils.resize(frame, width=self.RESIZE_WORKING_IMAGE)
            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]
                self.centroid_tracker.maxDistance = self.W / 8

            # convert to RGB color system for use with dlib
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # If saving file, save it here
            self.write_frame_to_video_file(file_to_export_video_to, frame)

            self.person_rectangles = []

            # Run the computationally expensive AI every few frames
            if self.total_frames % self.APPLY_AI_EVERY == 0:
                self.use_ai_to_find_things_in_frame(frame, rgb)

            else:  # Run on the "non AI processing" frames
                self.predict_path_of_things(rgb)

            # Add new centroids with guesses of where things will be
            centroid_loc_tracker_guesses = self.centroid_tracker.update(self.person_rectangles)

            # Draw or count each thing that is tracked
            for (objectID, centroid) in centroid_loc_tracker_guesses.items():
                # check to see if a trackable object exists for the current object ID
                _trackable_obj = self.trackable_objects.get(objectID, None)
                if _trackable_obj is None:
                    # Create new object if it doesn't exist
                    # TODO: Doesn't work when obj has been removed and loses an id in the table or extra
                    _detector = self.detector_types[objectID] if objectID in self.detector_types else self.detector_types[0]
                    _default_color = self.detector_info(_detector, 'color')
                    _label = "{} {}".format(_detector.title(), objectID)
                    _trackable_obj = TrackableObject(objectID, centroid, detector_type=_detector, label=_label,
                                                     default_color=_default_color, w=self.W, h=self.H)
                else:
                    # Add the new center and bounds to the existing object
                    _trackable_obj.add_centroid(centroid)
                    _trackable_obj.last_rect = self.closest_rect(centroid)

                    # Count if it meets some condition
                    self.count_trackable_object(_trackable_obj, centroid)

                # (re)store the trackable object in our dictionary
                self.trackable_objects[objectID] = _trackable_obj

                if self.viewer_show_lines:
                    self.show_tracker_lines_on_frame(_trackable_obj, frame, centroid)

            # Show all objects that are being tracked
            if self.viewer_show_boxes:
                self.show_tracker_boxes_on_frame(frame)

            # loop over the info tuples and draw them on our frame
            self.hud_info(True, frame)

            # Check to see if we should write the frame to disk
            if self.writer is not None:
                self.writer.write(frame)

            # Show the frame to screen
            cv2.imshow("Detection", frame)

            if paused:
                key = cv2.waitKey(-1)  # wait until any key is pressed
            else:
                key = cv2.waitKey(1) & 0xFF

            if key in [ord('p'), ord(' ')]:
                if paused:
                    paused = False
                else:
                    paused = True
                    key = cv2.waitKey(-1)  # wait until any key is pressed

            if key == ord("q"):  # User hit q
                user_let_finish = False
                break
            elif key == ord("l"):
                self.viewer_show_lines = not self.viewer_show_lines
            elif key == ord("b"):
                self.viewer_show_boxes = not self.viewer_show_boxes
            elif key == ord("n"):
                break
            elif key == ord('j'):
                self.write_paths_to_json()
            elif key == ord('c'):
                self.viewer_show_individual_colors = not self.viewer_show_individual_colors

            self.total_frames += 1
            self.fps.update()

        # Stop the timer and display FPS information
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        # loop over the info tuples and draw them on our frame
        self.hud_info(False)

        # Check to see if we need to release the video writer pointer
        if self.writer is not None:
            self.writer.release()

        # close the connection and close all windows
        stream.release()
        return user_let_finish

    def detector_info(self, detector_name, field, default=None):
        for d in self.DETECTORS:
            if d['detector_title'] == detector_name:
                if field in d:
                    return d[field]
        return default

    def write_paths_to_json(self, file_name="paths.json"):
        # Print out the points of the first path and save to file

        output_json = {
            "video_file": 'stream' if 'rtsp' in self.current_video_name else self.current_video_name,
            "frame_size": [self.W, self.H],
            "model": "Coco darknet face recognizer with dlib rgb backup",
            "nms_threshold": self.NMS_THRESHOLD,
            "ai_confidence_min": self.MIN_CONFIDENCE,
            "ai_applied_every": self.APPLY_AI_EVERY,
            "frames_before_disappearing_obj": self.FRAMES_BEFORE_DISAPPEAR,
            "distance_before_new_object": self.DISTANCE_BEFORE_NEW_OBJECT,
            "tracked_objects": []
        }
        trackable_json = []
        for _obj_key in self.trackable_objects.keys():
            _trackable_obj = self.trackable_objects[_obj_key]

            # Write out x and y lists (for use within matplotlib to terminal
            x_list = []
            y_list = []
            for count, lp in enumerate(_trackable_obj.centroids):
                x_list.append(int(lp[0]))
                y_list.append(int(lp[1]))
            print("x_list = {}".format(json.dumps(x_list)))
            print("y_list = {}".format(json.dumps(y_list)))

            output_obj = {
                "object_id": _obj_key,
                "detector_type": _trackable_obj.detector_type,
                "label": _trackable_obj.label,
                "centroids": [],
            }
            for count, lp in enumerate(_trackable_obj.centroids):
                output_obj["centroids"].append([int(lp[0]), int(lp[1])])

            trackable_json.append(output_obj)

        output_json['tracked_objects'] = trackable_json

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=2, separators=(',', ':'), cls=MyJSONEncoder)

    def show_tracker_boxes_on_frame(self, frame):
        for _obj_key in self.trackable_objects.keys():
            _trackable_obj = self.trackable_objects[_obj_key]

            _confidence = "0"
            _color = _trackable_obj.color if self.viewer_show_individual_colors else _trackable_obj.default_color
            _text = "{} [{}%]".format(_trackable_obj.label, _confidence)
            _r = _trackable_obj.last_rect
            if _r:
                if not type(_r) == tuple:  # If it's a dlib rect, pull out the rectangular values
                    _r = (_r.left(), _r.top(), _r.right(), _r.bottom())
                cv2.putText(frame, _text, (_r[0] - int(self.H / 70), _r[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _color, 2)
                _width = self.detector_info(_trackable_obj.detector_type, 'box_width', 2)
                cv2.rectangle(frame, (_r[0], _r[1]), (_r[2], _r[3]), _color, _width)
            # TODO: Verify this isn't used
            # else:
            # cv2.putText(frame, _text, (mid_x - int(self.H/70), mid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             _color, 2)

    def show_tracker_lines_on_frame(self, _trackable_obj, frame, centroid):
        mid_x = centroid[0]
        mid_y = centroid[1]

        lp_next = [mid_x, mid_y]
        for count, lp in enumerate(_trackable_obj.centroids):
            if count < len(_trackable_obj.centroids) - 1:
                lp_next = _trackable_obj.centroids[count + 1]
                _width = self.detector_info(_trackable_obj.detector_type, 'line_width', 1)
                cv2.line(frame, (int(lp[0]), int(lp[1])), (int(lp_next[0]), int(lp_next[1])),
                         _trackable_obj.color, _width)

        cv2.circle(frame, (lp_next[0], lp_next[1]), 4, (0, 255, 0), -1)

    def count_trackable_object(self, _trackable_obj, centroid):
        # the difference between the y-coordinate of the *current* centroid and the mean of *previous*
        # centroids tells us which direction the object is moving (negative for 'up' and pos for 'down')
        y = [c[1] for c in _trackable_obj.centroids]
        direction = centroid[1] - np.mean(y)

        # check to see if the object has been counted or not
        if not _trackable_obj.counted:  # TODO: Update for left, Right, multiple zones
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

    def use_ai_to_find_things_in_frame(self, frame, rgb):
        self.status = "Detecting"
        # TODO: Currently, we use a combo of person_trackers, detector_types, person_rectangles: these should be merged
        self.person_trackers = []
        self.detector_types = []

        # For each type of object being detected (e.g. person, car, dog) run through the neural detector
        for detector in self.DETECTORS:
            # Use the NN to find boxes that meet a minimum confidence
            layer_title = detector['detector_title']
            nn_person_layer_id = detector['model_layer_id']
            nn_detected_boxes = self.pedestrian_detection(frame, self.model, self.nn_layer_name,
                                                          model_layer_ids=nn_person_layer_id, layer_title=layer_title)

            for _box in nn_detected_boxes:
                start_x, start_y, end_x, end_y = int(_box[1][0]), int(_box[1][1]), int(_box[1][2]), int(_box[1][3])

                # Trim the box that the learning algorithm will focus on
                if detector['reduce_box']:
                    start_x, start_y, end_x, end_y = shorten_box(start_x, start_y, end_x, end_y)

                tracker = dlib.correlation_tracker()
                _rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                tracker.start_track(rgb, _rect)

                self.person_trackers.append(tracker)
                self.person_rectangles.append((start_x, start_y, end_x, end_y))
                self.detector_types.append(layer_title)

    def predict_path_of_things(self, rgb):
        # Use Dlib to Predict where all previous trackers say people will be from rgb
        self.status = "Tracking"
        for _track in self.person_trackers:
            # TODO: Add in dead reckoning prediction bounds of rgb instead of entire rgb
            _track.update(rgb)
            dlib_image_guess = _track.get_position()

            start_x, start_y = int(dlib_image_guess.left()), int(dlib_image_guess.top())
            end_x, end_y = int(dlib_image_guess.right()), int(dlib_image_guess.bottom())

            # update the rectangles of where things were seen
            self.person_rectangles.append((start_x, start_y, end_x, end_y))

    def closest_rect(self, centroid):
        closest = None
        dist_closest = 1000000

        for r in self.person_rectangles:  # TODO: Check type (car vs person, etc)
            cX = int((r[0] + r[2]) / 2.0)
            cY = int((r[1] + r[3]) / 2.0)

            dist = math.dist(centroid, [cX, cY])
            if dist < dist_closest:
                closest = r
                dist_closest = dist

        return closest

    def write_frame_to_video_file(self, file_to_export_video_to, frame):
        # Create the writer file and object if needed
        if not file_to_export_video_to == "" and self.writer is None:
            self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(file_to_export_video_to, self.fourcc, 30,
                                          (frame.shape[1], frame.shape[0]), True)

    def hud_info(self, show_on_cv_frame, frame=None):
        _hud_info = [
            ("People Seen", len(self.trackable_objects)),
            ("Going Up past center", self.total_up),
            ("Going Down past center", self.total_down),
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

        # DETECTORS = [
        #     {"detector_title": "person", "box_width": 4, "color": "random", "reduce_box": True},
        #     {"detector_title": "car", "box_width": 1, "color": "brown", "reduce_box": True},
        #     {"detector_title": "dog", "box_width": 2, "color": "white", "reduce_box": True},
        # ]

        # Add in coco darknet layer id
        for i, layer in enumerate(self.DETECTORS):
            self.DETECTORS[i]['model_layer_id'] = _coco_labels.index(layer['detector_title'])

        return layer_name, model

    def pedestrian_detection(self, image, model, layer_name, model_layer_ids=0, layer_title="person"):
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

                if class_id == model_layer_ids and confidence > self.MIN_CONFIDENCE:
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


def next_file_in_dir(current_file=None, directory="videos"):
    file_list = os.listdir(directory)
    next_index = 0
    if current_file in file_list:
        next_index = file_list.index(current_file) + 1
    if next_index >= len(file_list):
        next_index = 0
    return file_list[next_index]


def shorten_box(_l, _t, _r, _b):
    # return the top half of a box and middle 80%
    _h = _b - _t
    _w = _r - _l

    _b = int(_b - (_h * .5))
    _l = int(_l + (_w * .2))
    _r = int(_r - (_w * .2))

    return _l, _t, _r, _b


# ----------------------------
# Used to pretty print json from - https://stackoverflow.com/questions/13249415/
# how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module

class MyJSONEncoder(json.JSONEncoder):

    def iterencode(self, o, _one_shot=False):
        list_lvl = 0
        for s in super(MyJSONEncoder, self).iterencode(o, _one_shot=_one_shot):
            if s.startswith('['):
                list_lvl += 1
                s = s.replace('\n', '').rstrip()
                s = s.replace(' ', '')
            elif list_lvl > 1:
                s = s.replace('\n', '').rstrip()
                if s and s[-1] == ',':
                    s = s[:-1] + self.item_separator
                elif s and s[-1] == ':':
                    s = s[:-1] + self.key_separator
                s = s.replace(' ', '')
            if s.endswith(']'):
                list_lvl -= 1
            yield s


PD = PedestrianDetector()
