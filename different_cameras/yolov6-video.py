# coding=utf-8
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from utils import displayFrame, send_img

videoPath = r"demo.mp4"

numClasses = 80

blob = Path(__file__).parent.joinpath("yolov6n.blob")
model = dai.OpenVINO.Blob(blob)
dim = next(iter(model.networkInputs.values())).dims
W, H = dim[:2]

output_name, output_tenser = next(iter(model.networkOutputs.items()))
if "yolov6" in output_name:
    numClasses = output_tenser.dims[2] - 5
else:
    numClasses = output_tenser.dims[2] // 3 - 5
# fmt: off
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]
# fmt: on

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
xinFrame = pipeline.create(dai.node.XLinkIn)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutNN = pipeline.create(dai.node.XLinkOut)

xinFrame.setStreamName("inFrame")
xoutNN.setStreamName("detections")

# Network specific settings
detectionNetwork.setBlob(model)
detectionNetwork.setConfidenceThreshold(0.5)

# Yolo specific parameters
detectionNetwork.setNumClasses(numClasses)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([])
detectionNetwork.setAnchorMasks({})
detectionNetwork.setIouThreshold(0.3)

# Linking
xinFrame.out.link(detectionNetwork.input)
detectionNetwork.out.link(xoutNN.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Input queue will be used to send video frames to the device.
    inFrameQueue = device.getInputQueue(name="inFrame")
    # Output queue will be used to get nn data from the video frames.
    detectQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    frame = None
    detections = []
    # Random Colors for bounding boxes
    bboxColors = np.random.randint(256, size=(numClasses, 3)).tolist()


    cap = cv2.VideoCapture(videoPath)
    while cap.isOpened():
        read_correctly, frame = cap.read()
        if not read_correctly:
            break

        send_img(inFrameQueue, frame, W, H)

        detectQueueData = detectQueue.get()

        if detectQueueData is not None:
            detections = detectQueueData.detections

        if frame is not None:
            displayFrame(frame, detections, labelMap, bboxColors)
            cv2.imshow("image", frame)

        if cv2.waitKey(1) == ord("q"):
            break
