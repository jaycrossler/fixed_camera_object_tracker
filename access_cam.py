import numpy as np
import cv2
import os
import imutils
import dlib
from imutils.video import FPS

# NOTE: Install cmake, miniconda and dlib:
#   conda install -c conda-forge dlib


class PedestrianDetector:

    NMS_THRESHOLD = 0.3
    MIN_CONFIDENCE = 0.2
    RESIZE_WORKING_IMAGE = 700
    USE_CUDA_GPU = False
    EXPORT_TO_VIDEO = ""

    _coco_labels_path = "coco.names"
    _weights_path = "yolov4-tiny.weights"
    _config_path = "yolov4-tiny.cfg"
    # insert the HTTP(S)/RSTP feed from the camera
    # url = "rtsp://admin:PW@@192.168.1.62/Streaming/Channels/102"
    stream_url = "videos/annke2hd.20220529_235055_1.mp4"
    fps = None
    writer = None
    person_trackers = []
    tracker_labels = []
    fourcc = None

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
        self.tracker_labels = []

        # Start tracking video speed
        self.fps = FPS().start()
        while True:
            (grabbed, frame) = stream.read()
            if not grabbed:
                break

            # Make the image smaller to work with and convert to RGB color system
            frame = imutils.resize(frame, width=self.RESIZE_WORKING_IMAGE)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create the writer file and object if needed
            if not self.EXPORT_TO_VIDEO and self.writer is None:
                self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.writer = cv2.VideoWriter(self.EXPORT_TO_VIDEO, self.fourcc, 30,
                                              (frame.shape[1], frame.shape[0]), True)

            # Detect boxes around people
            pedestrian_boxes = self.pedestrian_detection(frame, model, layer_name, person_ids=person_layer_id)

            # Draw those boxes on the image
            for _box in pedestrian_boxes:
                cv2.rectangle(frame, (_box[1][0], _box[1][1]), (_box[1][2], _box[1][3]), (0, 255, 0), 2)

            # Check to see if we should write the frame to disk
            if self.writer is not None:
                self.writer.write(frame)

            # Show the frame to screen
            cv2.imshow("Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # User hit q
                break

        # Stop the timer and display FPS information
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        # Check to see if we need to release the video writer pointer
        if self.writer is not None:
            self.writer.release()

        # close the connection and close all windows
        stream.release()
        cv2.destroyAllWindows()

    def setup_model(self):
        _coco_labels = open(self._coco_labels_path).read().strip().split("\n")
        model = cv2.dnn.readNetFromDarknet(self._config_path, self._weights_path)
        if self.USE_CUDA_GPU:
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        layer_name = model.getLayerNames()  # TODO: Unused?
        layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

        person_layer_id = _coco_labels.index("person")

        return layer_name, model, person_layer_id

    def pedestrian_detection(self, image, model, layer_name, person_ids=0):
        (H, W) = image.shape[:2]
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

                    box = detection[0:4] * np.array([W, H, W, H])
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
