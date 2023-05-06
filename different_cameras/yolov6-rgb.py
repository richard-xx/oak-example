# coding=utf-8
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from utils import displayFrame

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
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
imageManip = pipeline.create(dai.node.ImageManip)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("image")
xoutNN.setStreamName("detections")

# Properties
# camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
# camRgb.setInterleaved(False)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(60)

imageManip.initialConfig.setResizeThumbnail(W, H, 114, 114, 114)
imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
imageManip.setMaxOutputFrameSize(W * H * 3)

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
camRgb.isp.link(imageManip.inputImage)

imageManip.out.link(detectionNetwork.input)
imageManip.out.link(xoutRgb.input)
detectionNetwork.out.link(xoutNN.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    imageQueue = device.getOutputQueue(name="image", maxSize=4, blocking=False)
    detectQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    frame = None
    detections = []
    # Random Colors for bounding boxes
    bboxColors = np.random.randint(256, size=(numClasses, 3)).tolist()

    while not device.isClosed():
        imageQueueData = imageQueue.tryGet()
        detectQueueData = detectQueue.tryGet()

        if imageQueueData is not None:
            frame = imageQueueData.getCvFrame()

        if detectQueueData is not None:
            detections = detectQueueData.detections

        if frame is not None:
            displayFrame(frame, detections, labelMap, bboxColors)
            # Show the frame
            cv2.imshow("image", frame)

        if cv2.waitKey(1) == ord("q"):
            break
