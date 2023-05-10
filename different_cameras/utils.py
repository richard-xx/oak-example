# coding=utf-8
from time import monotonic

import cv2
import depthai as dai
import numpy as np


def frameNorm(frame, bbox):
    """
    nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height

    :param frame:
    :param bbox:
    :return:
    """
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def drawText(frame, text, org, color=(255, 255, 255), thickness=1):
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA
    )


def drawRect(frame, topLeft, bottomRight, color=(255, 255, 255), thickness=1):
    cv2.rectangle(frame, topLeft, bottomRight, (0, 0, 0), thickness + 3)
    cv2.rectangle(frame, topLeft, bottomRight, color, thickness)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose((2, 0, 1)).flatten()


def send_img(inFrameQueue, frame, W, H):
    img = dai.ImgFrame()
    img.setData(to_planar(frame, (W, H)))
    img.setTimestamp(dai.Clock.now())
    img.setWidth(W)
    img.setHeight(H)
    inFrameQueue.send(img)
def displayFrame(frame, detections, labelMap, bboxColors, depthFrameColor=None):
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
            drawRect(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                bboxColors[detection.label]
            )
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

                drawText(
                    depthFrameColor,
                    labelMap[detection.label],
                    (xmin + 10, ymin + 20),
                )

                drawText(
                    depthFrameColor,
                    f"{detection.confidence:.2%}",
                    (xmin + 10, ymin + 35),
                )

                drawText(
                    depthFrameColor,
                    f"X: {int(detection.spatialCoordinates.x)} mm",
                    (xmin + 10, ymin + 50),
                )
                drawText(
                    depthFrameColor,
                    f"Y: {int(detection.spatialCoordinates.y)} mm",
                    (xmin + 10, ymin + 65),
                )
                drawText(
                    depthFrameColor,
                    f"Z: {int(detection.spatialCoordinates.z)} mm",
                    (xmin + 10, ymin + 80),
                )

                drawRect(
                    depthFrameColor,
                    (xmin, ymin),
                    (xmax, ymax),
                    bboxColors[detection.label],
                )

                drawText(
                    frame,
                    f"X: {int(detection.spatialCoordinates.x)} mm",
                    (bbox[0] + 10, bbox[1] + 50),
                )
                drawText(
                    frame,
                    f"Y: {int(detection.spatialCoordinates.y)} mm",
                    (bbox[0] + 10, bbox[1] + 65),
                )
                drawText(
                    frame,
                    f"Z: {int(detection.spatialCoordinates.z)} mm",
                    (bbox[0] + 10, bbox[1] + 80),
                )

