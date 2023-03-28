# coding=utf-8
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

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
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
stereo = pipeline.create(dai.node.StereoDepth)
imageManip = pipeline.create(dai.node.ImageManip)

xoutManip = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutManip.setStreamName("image")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
imageManip.initialConfig.setResize(W, H)
imageManip.setMaxOutputFrameSize(W * H * 3)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(
    dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_RIGHT
)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# Network specific settings
spatialDetectionNetwork.setBlob(model)
spatialDetectionNetwork.setConfidenceThreshold(0.5)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(numClasses)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([])
spatialDetectionNetwork.setAnchorMasks({})
spatialDetectionNetwork.setIouThreshold(0.3)

# spatial specific parameters
spatialDetectionNetwork.setBoundingBoxScaleFactor(1)
spatialDetectionNetwork.setDepthLowerThreshold(0)
spatialDetectionNetwork.setDepthUpperThreshold(50000)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

imageManip.out.link(spatialDetectionNetwork.input)
imageManip.out.link(xoutManip.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.rectifiedRight.link(imageManip.inputImage)
stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    imageQueue = device.getOutputQueue(name="image", maxSize=4, blocking=False)
    detectQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

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

    while not device.isClosed():
        imageQueueData = imageQueue.tryGet()
        detectQueueData = detectQueue.tryGet()
        depthQueueData = depthQueue.get()

        if imageQueueData is not None:
            frame = imageQueueData.getCvFrame()

        if detectQueueData is not None:
            detections = detectQueueData.detections

        if depthQueueData is not None:
            # depthFrame values are in millimeters
            depthFrame = depthQueueData.getFrame()
            depthFrameColor = cv2.normalize(
                depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
            )
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        if frame is not None:
            displayFrame("image", frame, depthFrameColor)

        if cv2.waitKey(1) == ord("q"):
            break
