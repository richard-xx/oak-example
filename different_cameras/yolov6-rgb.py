# coding=utf-8
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

blob = Path(__file__).parent.joinpath("yolov6n.blob")
model = dai.OpenVINO.Blob(blob)
dim = model.networkInputs.get("images").dims
W, H = dim[:2]
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
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("image")
xoutNN.setStreamName("detections")

# Properties
camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(60)

# Network specific settings
detectionNetwork.setBlob(model)
detectionNetwork.setConfidenceThreshold(0.5)

# Yolo specific parameters
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([])
detectionNetwork.setAnchorMasks({})
detectionNetwork.setIouThreshold(0.5)

# Linking
camRgb.preview.link(detectionNetwork.input)
camRgb.preview.link(xoutRgb.input)
detectionNetwork.out.link(xoutNN.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    imageQueue = device.getOutputQueue(name="image", maxSize=4, blocking=False)
    detectQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    frame = None
    detections = []
    color2 = (255, 255, 255)

    def frameNorm(frame, bbox):
        """
        nn data, being the bounding box locations, are in <0..1> range
        - they need to be normalized with frame width/height
        """
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, depthFrameColor=None):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )
            cv2.putText(
                frame,
                labelMap[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                f"{detection.confidence:.2%}",
                (bbox[0] + 10, bbox[1] + 35),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            if hasattr(detection, "boundingBoxMapping") and depthFrameColor is not None:
                roi = detection.boundingBoxMapping.roi
                roi = roi.denormalize(
                    depthFrameColor.shape[1], depthFrameColor.shape[0]
                )
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)
                cv2.rectangle(
                    depthFrameColor,
                    (xmin, ymin),
                    (xmax, ymax),
                    color,
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                )

                cv2.putText(
                    frame,
                    f"X: {int(detection.spatialCoordinates.x)} mm",
                    (bbox[0] + 10, bbox[1] + 50),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                cv2.putText(
                    frame,
                    f"Y: {int(detection.spatialCoordinates.y)} mm",
                    (bbox[0] + 10, bbox[1] + 65),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                cv2.putText(
                    frame,
                    f"Z: {int(detection.spatialCoordinates.z)} mm",
                    (bbox[0] + 10, bbox[1] + 80),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )

        # Show the frame
        cv2.imshow(name, frame)
        if depthFrameColor is not None:
            cv2.imshow("depth", depthFrameColor)

    while not device.isClosed():
        inRgb = imageQueue.tryGet()
        inDet = detectQueue.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

        if inDet is not None:
            detections = inDet.detections

        if frame is not None:
            displayFrame("image", frame)

        if cv2.waitKey(1) == ord("q"):
            break
