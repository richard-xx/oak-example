# OAK 使用不同相机进行模型推理

> 使用 oak 的 `LEFT`，`RIGHT` 和 `RGB` 相机进行 `YOLO` 检测

## **▌RGB**

> 使用 `RGB` 相机作为输入源

```python
...
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
...
camRgb.setPreviewSize(W, H)
...
camRgb.preview.link(detectionNetwork.input)
...
```

详见：[yolov6-rgb.py](yolov6-rgb.py)

## **▌RGB + DEPTH**

> 使用 `RGB` 相机作为输入源，并附加深度信息

```python
...
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
...
camRgb.setPreviewSize(W, H)
...
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
...
# 将深度图与 RGB 相机的视角对齐，在其上进行推理
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
...
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
stereo.depth.link(spatialDetectionNetwork.inputDepth)
...
```

详见：[yolov6-rgb-spatial.py](yolov6-rgb-spatial.py)

---

## **▌RIGHT**

> 使用 `RIGHT` 相机作为输入源

```python
...
monoRight = pipeline.create(dai.node.MonoCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
imageManip = pipeline.create(dai.node.ImageManip)
...
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
...
# NN 模型需要 BGR 输入。默认情况下 ImageManip 输出类型与输入相同（在本例中为灰色）
imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
imageManip.initialConfig.setResize(W, H)
imageManip.setMaxOutputFrameSize(W * H * 3)
...
monoRight.out.link(imageManip.inputImage)
imageManip.out.link(detectionNetwork.input)
...
```

详见：[yolov6-right.py](yolov6-right.py)

## **▌RIGHT + DEPTH**

> 使用 `RIGHT` 相机作为输入源，并附加深度信息

```python
...
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
stereo = pipeline.create(dai.node.StereoDepth)
imageManip = pipeline.create(dai.node.ImageManip)
...
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
...
# NN 模型需要 BGR 输入。默认情况下 ImageManip 输出类型与输入相同（在本例中为灰色）
imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
imageManip.initialConfig.setResize(W, H)
imageManip.setMaxOutputFrameSize(W * H * 3)
...
# 将深度图与 RIGHT 相机的视角对齐，在其上进行推理
stereo.setDepthAlign(
    dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_RIGHT
)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
...
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

imageManip.out.link(spatialDetectionNetwork.input)

stereo.rectifiedRight.link(imageManip.inputImage)
stereo.depth.link(spatialDetectionNetwork.inputDepth)
...
```

详见：[yolov6-right-spatial.py](yolov6-right-spatial.py)

---

## **▌LEFT**

> 使用 `LEFT` 相机作为输入源

```python
...
monoLeft = pipeline.create(dai.node.MonoCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
imageManip = pipeline.create(dai.node.ImageManip)
...
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
...
# NN 模型需要 BGR 输入。默认情况下 ImageManip 输出类型与输入相同（在本例中为灰色）
imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
imageManip.initialConfig.setResize(W, H)
imageManip.setMaxOutputFrameSize(W * H * 3)
...
monoLeft.out.link(imageManip.inputImage)
imageManip.out.link(detectionNetwork.input)
...
```

详见：[yolov6-left.py](yolov6-left.py)

## **▌LEFT + DEPTH**

> 使用 `LEFT` 相机作为输入源，并附加深度信息

```python
...
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
stereo = pipeline.create(dai.node.StereoDepth)
imageManip = pipeline.create(dai.node.ImageManip)
...
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
...
# NN 模型需要 BGR 输入。默认情况下 ImageManip 输出类型与输入相同（在本例中为灰色）
imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
imageManip.initialConfig.setResize(W, H)
imageManip.setMaxOutputFrameSize(W * H * 3)
...
# 将深度图与 LEFT 相机的视角对齐，在其上进行推理
stereo.setDepthAlign(
    dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
...
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

imageManip.out.link(spatialDetectionNetwork.input)

stereo.rectifiedLeft.link(imageManip.inputImage)
stereo.depth.link(spatialDetectionNetwork.inputDepth)
...
```

详见：[yolov6-left-spatial.py](yolov6-left-spatial.py)

## **▌VIDEO**

> 使用 `VIDEO` 作为输入源

```python
...
xinFrame = pipeline.create(dai.node.XLinkIn)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
...
xinFrame.setStreamName("inFrame")
...
xinFrame.out.link(detectionNetwork.input)
...
# 输入队列将用于将视频帧发送到设备。
inFrameQueue = device.getInputQueue(name="inFrame")
...
img = dai.ImgFrame()
img.setData(to_planar(frame, (W, H)))
img.setTimestamp(monotonic())
img.setWidth(W)
img.setHeight(H)
inFrameQueue.send(img)
...
```

详见：[yolov6-video.py](yolov6-video.py)