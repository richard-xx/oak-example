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
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
imageManip = pipeline.create(dai.node.ImageManip)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("image")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")

# Properties
# camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
# camRgb.setInterleaved(False)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(60)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

imageManip.initialConfig.setResizeThumbnail(W, H, 114, 114, 114)
imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
imageManip.setMaxOutputFrameSize(W * H * 3)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
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

camRgb.isp.link(imageManip.inputImage)

imageManip.out.link(spatialDetectionNetwork.input)
imageManip.out.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

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
    # Random Colors for bounding boxes
    bboxColors = np.random.randint(256, size=(numClasses, 3)).tolist()

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
            displayFrame(frame, detections, labelMap, bboxColors, depthFrameColor)
            # Show the frame
            cv2.imshow("image", frame)
            cv2.imshow("depth", depthFrameColor)
        if cv2.waitKey(1) == ord("q"):
            break
