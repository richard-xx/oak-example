# coding=utf-8
from pathlib import Path
from time import monotonic

import cv2
import depthai as dai
import numpy as np

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
    bboxColors = (
            np.random.random(size=(256, 3)) * numClasses
    )  # Random Colors for bounding boxes

    def frameNorm(frame, bbox):
        """
        nn data, being the bounding box locations, are in <0..1> range
        - they need to be normalized with frame width/height
        """
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    def drawText(frame, text, org, color=(255, 255, 255)):
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    def displayFrame(name, frame, depthFrameColor=None):
        for detection in detections:
            bbox = frameNorm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )
            drawText(
                frame,
                labelMap[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
            )
            drawText(
                frame,
                f"{detection.confidence:.2%}",
                (bbox[0] + 10, bbox[1] + 35),
            )
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bboxColors[detection.label], 2)
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
                    bboxColors[detection.label],
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                )

                drawText(
                    depthFrameColor,
                    f"X: {int(detection.spatialCoordinates.x)} mm",
                    (bbox[0] + 10, bbox[1] + 50),
                )
                drawText(
                    depthFrameColor,
                    f"Y: {int(detection.spatialCoordinates.y)} mm",
                    (bbox[0] + 10, bbox[1] + 65),
                )
                drawText(
                    depthFrameColor,
                    f"Z: {int(detection.spatialCoordinates.z)} mm",
                    (bbox[0] + 10, bbox[1] + 80),
                )

        # Show the frame
        cv2.imshow(name, frame)
        if depthFrameColor is not None:
            cv2.imshow("depth", depthFrameColor)

    cap = cv2.VideoCapture(videoPath)
    while cap.isOpened():
        read_correctly, frame = cap.read()
        if not read_correctly:
            break

        img = dai.ImgFrame()
        img.setData(to_planar(frame, (W, H)))
        img.setTimestamp(monotonic())
        img.setWidth(W)
        img.setHeight(H)
        inFrameQueue.send(img)

        detectQueueData = detectQueue.get()

        if detectQueueData is not None:
            detections = detectQueueData.detections

        if frame is not None:
            displayFrame("image", frame)

        if cv2.waitKey(1) == ord("q"):
            break
