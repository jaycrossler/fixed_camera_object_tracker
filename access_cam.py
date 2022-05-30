import numpy as np
import cv2
import os
import imutils

# NOTE: Install cmake, miniconda and dlib:
#   conda install -c conda-forge dlib


class PedestrianDetector:

    NMS_THRESHOLD = 0.3
    MIN_CONFIDENCE = 0.2
    USE_CUDA_GPU = False
    _coco_labels_path = "coco.names"
    _weights_path = "yolov4-tiny.weights"
    _config_path = "yolov4-tiny.cfg"
    # insert the HTTP(S)/RSTP feed from the camera
    # url = "rtsp://admin:PalantirJinx42@@192.168.1.62/Streaming/Channels/102"
    stream_url = "videos/annke2hd.20220529_235055_1.mp4"

    def __init__(self):
        self.initialize()

    def initialize(self):

        layer_name, model, person_layer_id = self.setup_model()

        # open the video
        current_frame = cv2.VideoCapture(self.stream_url)
        while True:
            (grabbed, image) = current_frame.read()

            if not grabbed:
                break
            image = imutils.resize(image, width=700)
            pedestrian_boxes = self.pedestrian_detection(image, model, layer_name, person_ids=person_layer_id)

            for _box in pedestrian_boxes:
                cv2.rectangle(image, (_box[1][0], _box[1][1]), (_box[1][2], _box[1][3]), (0, 255, 0), 2)

            cv2.imshow("Detection", image)

            key = cv2.waitKey(1)
            if key == 27:
                break

        # close the connection and close all windows
        current_frame.release()
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
