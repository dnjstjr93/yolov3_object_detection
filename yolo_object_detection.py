__author__  = 'Wonseok Jung'
__date__    = '2021/06/06'

import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="YOLO Object Detection")
parser.add_argument("--image", type=str, default="data/room_ser.jpg", help="image source.")
parser.add_argument("--imageSize", default=416, type=int,
                    help="image size")
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
args = parser.parse_args()

# Load Yolo
modelConfiguration = args.config_file
modelWeights = args.weight_file

# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net = cv2.dnn.readNet(modelWeights, modelConfiguration)
classes = []
with open(args.class_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread(args.image)
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (args.imageSize, args.imageSize), (0, 0, 0), True, crop=False)

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
        if confidence > args.confidence_threshold:
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

indexes = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence_threshold, args.nms_threshold)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
