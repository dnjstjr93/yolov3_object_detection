# import cv2
# import os
# import argparse
# import sys
# import numpy as np
# import time
#
# parser = argparse.ArgumentParser(description="YOLO Object Detection")
# parser.add_argument("--image", type=str,
#                     help="image source.")
# parser.add_argument("--video", type=str, default="Scooters-5638.mp4",
#                     help="video source.")
# args = parser.parse_args()
#
# print(args)
#
# # Initialize the parameters
# confThreshold = 0.5
# confThreshold = 0.4
# inpWidth = 416
# inpHeight = 416
#
# # video_path = "traffic.mp4"
# video_path = "Scooters-5638.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
#
# # Load names of classes
# classesFile = "coco.names"
# classes = None
# with open(classesFile, 'rt') as f:
#     classes = f.read().rstrip('\n').split('\n')
#
# # Give the configuration and weight files for the model and load the network using them.
# modelConfiguration = "yolov3.cfg"
# modelWeights = "yolov3.weights"
#
# # net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#
#
# # Get the names of the output layers
# def getOutputsNames(net):
#     # Get the names of all the layers in the network
#     layersNames = net.getLayerNames()
#     # Get the names of the output layers, i. e. the layers with unconnected outputs
#     return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#
#
# # Remove the bounding boxes with low confidence using non-maxima suppression
# def postprocess(frame, outs):
#     frameHeight = frame.shape[0]
#     frameWidth = frame.shape[1]
#
#     classIds = []
#     confidences = []
#     boxes = []
#     # Scan through all the bounding boxes output from the network and keep only the
#     # ones with high confidence scores. Assign the box's class label as the class with the highest score.
#     print(outs)
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classIds]
#             if confidence > confThreshold:
#                 # Object detected
#                 center_x = int(detection[0] * frameWidth)
#                 center_y = int(detection[1] * frameHeight)
#                 width = int(detection[2] * frameWidth)
#                 height = int(detection[3] * frameHeight)
#
#                 # Rectangle coordinates
#                 x = int(center_x - width / 2)
#                 y = int(center_y - height / 2)
#
#                 boxes.append([x, y, width, height])
#                 confidences.append(float(confidence))
#                 classIds.append(classId)
#     # Perform non maximum suppression to eliminate redundant overlapping boxes with
#     # lower confidences.
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
#     for i in indices:
#         i = i[0]
#         box = boxes[i]
#         x = box[0]
#         y = box[1]
#         width = box[2]
#         height = box[3]
#         drawPred(classIds[i], confidences[i], x, y, x + width, y + height)
#
#
# # Draw the predicted bounding box
# def drawPred(classId, conf, left, top, right, bottom):
#     # Draw a bounding box.
#     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))
#
#     label = '%.2f' % conf
#
#     # Get the label for the class name and its confidence
#     if classes:
#         assert (classId < len(classes))
#         lebel = '%s:%s' % (classes[classId], label)
#
#     # Display the label at the top of the bounding box
#     labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#     top = max(top, labelSize[1])
#     cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
#
#
# if args.image:
#     # Open the image file
#     if not os.path.isfile(args.image):
#         print("Input image file", args.image, " doesn't exist")
#         sys.exit(1)
#     cap = cv2.VideoCapture(args.image)
#     outputFile = args.image[:-4]+'_yolo_out_py.jpg'
# elif args.video:
#     # Open the video file
#     if not os.path.isfile(args.video):
#         print("Input Video file", args.video, " doesn't exist")
#         sys.exit(1)
#     cap = cv2.VideoCapture(args.video)
#     outputFile = args.video[:-4]+'_yolo_out_py.mp4'
# else:
#     # Webcam input
#     cap = cv2.VideoCapture(0)
#
# # Get the video writer initialized to save the output video
# if not args.image:
#     vid_writer = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
#
# while cv2.waitKey(1) < 0:
#     # get frame from the video
#     hasFrame, frame = cap.read()
#
#     # Stop the program if reached end of video
#     if not hasFrame:
#         print("Done processing !!!")
#         print("Output file is stored as ", outputFile)
#         cv2.waitKey(3000)
#         break
#
#     # Create a 4D blob from a frame.
#     blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], True, crop=False)
#
#     # Set the input to the network
#     net.setInput(blob)
#
#     # Runs the forward pass to get output of the output layers
#     outs = net.forward(getOutputsNames(net))
#
#
#     # Remove the bounding boxes with low confidence
#     postprocess(frame, outs)
#
#     # Put efficiency information. The function getPerfProfile returns the
#     # overall time for inference(t) and the timings for each of the layers(in layersTimes)
#     t, _ = net.getPerfProfile()
#     label = 'Inference time: %.2f ms' %(t * 1000.0 / cv2.getTickFrequency())
#     cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
#
#     # Writer the frame with the detection boxes
#     if (args.image):
#         cv2.imwrite(outputFile, frame.astype(np.uint8))
#     else:
#         cv2.imshow('result', frame)
#         # vid_writer.write(frame.astype(np.uint8))