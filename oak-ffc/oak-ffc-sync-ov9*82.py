# coding=utf-8
from __future__ import annotations

import collections
import time

import cv2
import depthai as dai

FPS = 60
FRAME_SYNC_OUTPUT = "CAM_A"
"""
# OV9782
cam_list = {
    "CAM_A": {"color": True, "res": "800"},
    "CAM_B": {"color": True, "res": "800"},
    "CAM_C": {"color": True, "res": "800"},
    "CAM_D": {"color": True, "res": "800"},
}
"""

# OV9282
cam_list = {
    "CAM_A": {"color": False, "res": "400"},
    "CAM_B": {"color": False, "res": "400"},
    "CAM_C": {"color": False, "res": "400"},
    "CAM_D": {"color": False, "res": "400"},
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
    "CAM_A": "CAM_A",
    "CAM_B": "CAM_B",
    "CAM_C": "CAM_C",
    "CAM_D": "CAM_D",
}

cam_socket_opts = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
}


def create_pipeline():
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
        cam[cam_name].setFps(FPS)
        # cam[cam_name].initialControl.setExternalTrigger(4, 3)

        if cam_name == FRAME_SYNC_OUTPUT:
            cam[cam_name].initialControl.setFrameSyncMode(
                dai.CameraControl.FrameSyncMode.OUTPUT
            )
        else:
            cam[cam_name].initialControl.setFrameSyncMode(
                dai.CameraControl.FrameSyncMode.INPUT
            )
    return pipeline


def main():
    global cam_list

    # 创建 DepthAI 设备配置对象
    config = dai.Device.Config()

    # 设置 GPIO 引脚 6 为输出模式，初始状态为高电平
    config.board.gpio[6] = dai.BoardConfig.GPIO(
        dai.BoardConfig.GPIO.OUTPUT, dai.BoardConfig.GPIO.Level.HIGH
    )
    # 设置 OpenVINO 版本号
    # config.version = dai.OpenVINO.VERSION_2021_4

    # 创建 DepthAI 设备对象
    with dai.Device(config) as device:
        # 获取连接到设备上的相机列表，输出相机名称、分辨率、支持的颜色类型等信息
        print("Connected cameras:")
        sensor_names = {}  # type: dict[str, str]
        for p in device.getConnectedCameraFeatures():
            # 输出相机信息
            print(
                f" -socket {p.socket.name:6}: {p.sensorName:6} {p.width:4} x {p.height:4} focus:",
                end="",
            )
            print("auto " if p.hasAutofocus else "fixed", "- ", end="")
            supported_types = [color_type.name for color_type in p.supportedTypes]
            print(*supported_types)

            # 更新相机属性表
            cam_name = cam_socket_to_name[p.socket.name]
            sensor_names[cam_name] = p.sensorName

        # 仅保留设备已连接的相机
        for cam_name in set(cam_list).difference(sensor_names):
            print(f"{cam_name} is not connected !")

        cam_list = {
            name: cam_list[name] for name in set(cam_list).intersection(sensor_names)
        }

        # 开始执行给定的管道
        device.startPipeline(create_pipeline())

        # 创建相机输出队列
        output_queues = {}
        for cam_name in cam_list:
            output_queues[cam_name] = device.getOutputQueue(
                name=cam_name, maxSize=4, blocking=False
            )
            cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam_name, 640, 480)

        fps_handler = FPSHandler()

        # 循环读取并显示视频流
        while not device.isClosed():
            frame_list = []
            for cam_name in cam_list:
                packet = output_queues[cam_name].tryGet()
                if packet is not None:
                    # 输出视频帧的时间戳
                    print(cam_name + ":", packet.getTimestampDevice())
                    # 获取视频帧并添加到帧列表中
                    frame_list.append((cam_name, packet.getCvFrame()))

            if frame_list:
                fps_handler.tick("FRAME")
                print("-------------------------------")
                # 显示视频帧
                for cam_name, frame in frame_list:
                    fps_handler.draw_fps(frame, "FRAME")
                    cv2.imshow(cam_name, frame)

            # 等待用户按下 "q" 键，退出循环并关闭窗口
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

class FPSHandler:
    """
    Class that handles all FPS-related operations.

    Mostly used to calculate different streams FPS, but can also be
    used to feed the video file based on its FPS property, not app performance (this prevents the video from being sent
    to quickly if we finish processing a frame earlier than the next video frame should be consumed)
    """

    _fps_bg_color = (0, 0, 0)
    _fps_color = (255, 255, 255)
    _fps_type = cv2.FONT_HERSHEY_SIMPLEX
    _fps_line_type = cv2.LINE_AA

    def __init__(self, cap=None, max_ticks=100):
        """
        Constructor that initializes the class with a video file object and a maximum ticks amount for FPS calculation

        Args:
            cap (cv2.VideoCapture, Optional): handler to the video file object
            max_ticks (int, Optional): maximum ticks amount for FPS calculation
        """
        self._timestamp = None
        self._start = None
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self._useCamera = cap is None

        self._iterCnt = 0
        self._ticks = {}

        if max_ticks < 2:  # noqa: PLR2004
            msg = f"Proviced max_ticks value must be 2 or higher (supplied: {max_ticks})"
            raise ValueError(msg)

        self._maxTicks = max_ticks

    def next_iter(self):
        """Marks the next iteration of the processing loop. Will use `time.sleep` method if initialized with video file object"""
        if self._start is None:
            self._start = time.monotonic()

        if not self._useCamera and self._timestamp is not None:
            frame_delay = 1.0 / self._framerate
            delay = (self._timestamp + frame_delay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._timestamp = time.monotonic()
        self._iterCnt += 1

    def tick(self, name):
        """
        Marks a point in time for specified name

        Args:
            name (str): Specifies timestamp name
        """
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tick_fps(self, name):
        """
        Calculates the FPS based on specified name

        Args:
            name (str): Specifies timestamps' name

        Returns:
            float: Calculated FPS or `0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            time_diff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / time_diff if time_diff != 0 else 0.0
        return 0.0

    def fps(self):
        """
        Calculates FPS value based on `nextIter` calls, being the FPS of processing loop

        Returns:
            float: Calculated FPS or `0.0` (default in case of failure)
        """
        if self._start is None or self._timestamp is None:
            return 0.0
        time_diff = self._timestamp - self._start
        return self._iterCnt / time_diff if time_diff != 0 else 0.0

    def print_status(self):
        """Prints total FPS for all names stored in :func:`tick` calls"""
        print("=== TOTAL FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tick_fps(name):.1f}")

    def draw_fps(self, frame, name):
        """
        Draws FPS values on requested frame, calculated based on specified name

        Args:
            frame (numpy.ndarray): Frame object to draw values on
            name (str): Specifies timestamps' name
        """
        frame_fps = f"{name.upper()} FPS: {round(self.tick_fps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        draw_text(
            frame,
            frame_fps,
            (5, 15),
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
        )

        if "nn" in self._ticks:
            draw_text(
                frame,
                f"NN FPS:  {round(self.tick_fps('nn'), 1)}",
                (5, 30),
                color=(255,255,255),
                bg_color=(0, 0, 0),
            )


def draw_text(
    frame,
    text,
    org,
    color=(255, 255, 255),
    bg_color=(128, 128, 128),
    font_scale=0.5,
    thickness=1,
):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, bg_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    main()
