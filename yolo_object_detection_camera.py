__author__  = 'Wonseok Jung'
__date__    = '2021/06/06'

import cv2
import numpy as np
import time
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="YOLO Object Detection")
parser.add_argument("--camera", type=str, default=int(0), help="camera source.")  # rtsp://admin:1234556@10.252.29.21/stream0
parser.add_argument("--cameraSize", default=416, type=int,
                    help="camera size")
parser.add_argument("--confidence_threshold", type=float, default=0.5,
                    help="threshold of confidence")
parser.add_argument("--nms_threshold", type=float, default=0.4,
                    help="remove detections with lower confidence")
parser.add_argument("--config_file", type=str, default="cfg/yolov3.cfg",
                    help="path to config file")
parser.add_argument("--weight_file", type=str, default="weights/yolov3.weights",
                    help="path to weights")
parser.add_argument("--class_file", type=str, default="cfg/coco.names",
                    help="path to classes file")
parser.add_argument("--fps", type=float, default=10.0,
                    help="FPS to output file")
args = parser.parse_args()

# Load Yolo
modelConfiguration = args.config_file
modelWeights = args.weight_file

net = cv2.dnn.readNet(modelWeights, modelConfiguration)

classes = []
with open(args.class_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

confThreshold = args.confidence_threshold
nmsThreshold = args.nms_threshold
inpWidth = args.cameraSize
inpHeight = args.cameraSize

# Loading camera
camera_path = args.camera
# # camera_path = "data/traffic.mp4"
cap = cv2.VideoCapture(camera_path)

# define output
now = datetime.now()
date = str(now)[0:10]
hour = (now.hour)
second = (now.second)
now_time = datetime.now().strftime("%d_%H-%M-%S")
path = ".\\output"
if not os.path.isdir(path):
    os.mkdir(path)
outputFile = 'output\\camera' + str(now_time) + '_yolov3_output.mp4'
FPS = args.fps
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(outputFile, fourcc, FPS, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

start = time.time()

while cv2.waitKey(1) < 0:
    ret, img = cap.read()

    if ret:
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (inpWidth, inpHeight), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confThreshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, confThreshold)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

        cv2.imshow('camera'+str(camera_path), img)
        video.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('error')

end = time.time()
print(end - start)
video.release()
cap.release()
cv2.destroyAllWindows()
