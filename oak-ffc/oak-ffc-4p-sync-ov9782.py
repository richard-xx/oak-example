# coding=utf-8
import time
from datetime import datetime

import cv2
import depthai as dai

cam_list = {
    "CAM_A": {"color": True, "res": "800"},
    "CAM_B": {"color": True, "res": "800"},
    "CAM_C": {"color": True, "res": "800"},
    "CAM_D": {"color": True, "res": "800"},
}

mono_res_opts = {
    "400": dai.MonoCameraProperties.SensorResolution.THE_400_P,
    "480": dai.MonoCameraProperties.SensorResolution.THE_480_P,
    "720": dai.MonoCameraProperties.SensorResolution.THE_720_P,
    "800": dai.MonoCameraProperties.SensorResolution.THE_800_P,
    "1200": dai.MonoCameraProperties.SensorResolution.THE_1200_P,
}

color_res_opts = {
    "720": dai.ColorCameraProperties.SensorResolution.THE_720_P,
    "800": dai.ColorCameraProperties.SensorResolution.THE_800_P,
    "1080": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    "1200": dai.ColorCameraProperties.SensorResolution.THE_1200_P,
    "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
    "5mp": dai.ColorCameraProperties.SensorResolution.THE_5_MP,
    "12mp": dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    "48mp": dai.ColorCameraProperties.SensorResolution.THE_48_MP,
}

cam_socket_to_name = {
    "RGB": "CAM_A",
    "LEFT": "CAM_B",
    "RIGHT": "CAM_C",
    "CAM_D": "CAM_D",
}

cam_socket_opts = {
    "CAM_A": dai.CameraBoardSocket.RGB,  # Or CAM_A
    "CAM_B": dai.CameraBoardSocket.LEFT,  # Or CAM_B
    "CAM_C": dai.CameraBoardSocket.RIGHT,  # Or CAM_C
    "CAM_D": dai.CameraBoardSocket.CAM_D,
}

pipeline = dai.Pipeline()
cam = {}
xout = {}
for cam_name, cam_props in cam_list.items():
    xout[cam_name] = pipeline.create(dai.node.XLinkOut)
    xout[cam_name].setStreamName(cam_name)
    if cam_props["color"]:
        cam[cam_name] = pipeline.create(dai.node.ColorCamera)
        cam[cam_name].setResolution(color_res_opts[cam_props["res"]])
        cam[cam_name].isp.link(xout[cam_name].input)
    else:
        cam[cam_name] = pipeline.createMonoCamera()
        cam[cam_name].setResolution(mono_res_opts[cam_props["res"]])
        cam[cam_name].out.link(xout[cam_name].input)
    cam[cam_name].setBoardSocket(cam_socket_opts[cam_name])
    # cam[cam_name].initialControl.setExternalTrigger(4, 3)
    cam[cam_name].setFps(20.0)

    if cam_name == "CAM_A":
        cam[cam_name].initialControl.setFrameSyncMode(
            dai.CameraControl.FrameSyncMode.OUTPUT
        )
    else:
        cam[cam_name].initialControl.setFrameSyncMode(
            dai.CameraControl.FrameSyncMode.INPUT
        )

config = dai.Device.Config()
config.board.gpio[6] = dai.BoardConfig.GPIO(
    dai.BoardConfig.GPIO.OUTPUT, dai.BoardConfig.GPIO.Level.HIGH
)
config.version = dai.OpenVINO.VERSION_2021_4

with dai.Device(config) as device:
    device.startPipeline(pipeline)

    print("Connected cameras:")
    cam_names = {}
    for p in device.getConnectedCameraFeatures():
        print(
            f" -socket {p.socket.name:6}: {p.sensorName:6} {p.width:4} x {p.height:4} focus:",
            end="",
        )
        print("auto " if p.hasAutofocus else "fixed", "- ", end="")
        print(*[color_type.name for color_type in p.supportedTypes])
        cam_names[cam_socket_to_name[p.socket.name]] = p.sensorName

    cam_list = {name: cam_list[name] for name in cam_list if name in cam_names.keys()}

    q = {}
    for cam_name in cam_list:
        q[cam_name] = device.getOutputQueue(name=cam_name, maxSize=4, blocking=True)
        cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam_name, 640, 480)

    # 用于存储需要截图的摄像头名称列表，按下 'c' 键可以切换
    capture_cam_list = []

    while not device.isClosed():
        # 存储当前采集到的每个摄像头最新的一帧图像
        frames = []

        # 遍历所有摄像头，从队列中获取最新的一帧图像
        for cam_name in cam_list:
            packet = q[cam_name].tryGet()  # type: dai.ImgFrame | None
            if packet is not None:
                print(cam_name + ":", packet.getTimestampDevice())
                frames.append((cam_name, packet.getCvFrame()))

        # 如果至少有一个摄像头有新的图像，则显示这些图像并保存截图（如果需要）
        if frames:
            print("-------------------------------")
            capture_time = datetime.now().strftime("%Y%m%d_%H%M%S.%f")

            for cam_name, frame in frames:
                cv2.imshow(cam_name, frame)
                # 如果该摄像头需要进行截图，则生成截图文件名并保存图片
                if cam_name in capture_cam_list:
                    width, height = frame.shape[:2]

                    capture_file_name = (
                        f"capture_{cam_name}_{cam_names[cam_name]}"
                        f"_{width}x{height}_"
                        f"{capture_time}.png"
                    )

                    print("Saving:", capture_file_name)
                    # 将图像编码为 PNG 格式并写入文件
                    image_data = cv2.imencode(".png", frame)[1]
                    image_data.tofile(capture_file_name)

                    # cv2.imwrite(capture_file_name, frame)
                    # capture_cam_list.remove(cam_name)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("c"):
            if capture_cam_list:
                capture_cam_list = []
            else:
                capture_cam_list = cam_list.copy()

    cv2.destroyAllWindows()
