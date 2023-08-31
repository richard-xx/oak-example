# coding=utf-8
from __future__ import annotations

import contextlib

import cv2
import depthai as dai

FRAME_SYNC_OUTPUT = "CAM_A"

cam_list = {
    "CAM_A": {"color": True, "res": "1080", "codec": "h265"},
    "CAM_B": {"color": False, "res": "800", "codec": "h265"},
    "CAM_C": {"color": False, "res": "800", "codec": "h265"},
    "CAM_D": {"color": True, "res": "1080", "codec": "h265"},
}

codec_opts = {
    "h264": dai.VideoEncoderProperties.Profile.H264_HIGH,
    "h265": dai.VideoEncoderProperties.Profile.H265_MAIN,
    "mjpeg": dai.VideoEncoderProperties.Profile.MJPEG,
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
    "CAM_E": "CAM_E",
    "CAM_F": "CAM_F",
}

cam_socket_opts = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
    "CAM_E": dai.CameraBoardSocket.CAM_E,
    "CAM_F": dai.CameraBoardSocket.CAM_F,
}


def create_pipeline():
    pipeline = dai.Pipeline()
    cam = {}
    xout = {}
    video_encoders = {}
    ve_out = {}
    for cam_name, cam_props in cam_list.items():
        xout[cam_name] = pipeline.create(dai.node.XLinkOut)
        xout[cam_name].setStreamName(cam_name)
        video_encoders[cam_name] = pipeline.create(dai.node.VideoEncoder)
        ve_out[cam_name] = pipeline.create(dai.node.XLinkOut)
        ve_out[cam_name].setStreamName(f"{cam_name}_ve")
        video_encoders[cam_name].bitstream.link(ve_out[cam_name].input)

        if cam_props["color"]:
            cam[cam_name] = pipeline.create(dai.node.ColorCamera)
            cam[cam_name].setResolution(color_res_opts[cam_props["res"]])
            cam[cam_name].isp.link(xout[cam_name].input)
            cam[cam_name].video.link(video_encoders[cam_name].input)
        else:
            cam[cam_name] = pipeline.createMonoCamera()
            cam[cam_name].setResolution(mono_res_opts[cam_props["res"]])
            cam[cam_name].out.link(xout[cam_name].input)
            cam[cam_name].out.link(video_encoders[cam_name].input)
        cam[cam_name].setBoardSocket(cam_socket_opts[cam_name])
        cam[cam_name].setFps(20.0)

        video_encoders[cam_name].setDefaultProfilePreset(
            cam[cam_name].getFps(), codec_opts[cam_props["codec"]]
        )

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
    config.version = dai.OpenVINO.VERSION_2021_4

    with contextlib.ExitStack() as stack:
        # 创建 DepthAI 设备对象
        device = stack.enter_context(dai.Device(config))

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
            cam_feature = cam_list.get(cam_name)
            if cam_feature:
                color_type = "COLOR" if cam_feature.get("color") else "MONO"
                if color_type not in supported_types:
                    cam_feature["color"] = not cam_feature["color"]

            sensor_names[cam_name] = p.sensorName

        # 仅保留设备已连接的相机
        for cam_name in set(cam_list).difference(sensor_names):
            print(f"{cam_name} is not connected !")

        cam_list = {
            name: cam_list[name] for name in set(cam_list).intersection(sensor_names)
        }

        # 开始执行给定的管道
        device.startPipeline(create_pipeline())

        # 创建相机输出队列和视频文件
        output_queues = {}
        video_queues = {}
        video_files = {}
        for cam_name in cam_list:
            output_queues[cam_name] = device.getOutputQueue(
                name=cam_name, maxSize=4, blocking=False
            )
            video_queues[cam_name] = device.getOutputQueue(
                name=f"{cam_name}_ve", maxSize=30, blocking=True
            )
            cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam_name, 640, 480)

            video_files[cam_name] = stack.enter_context(
                open(f"{cam_name}.{cam_list[cam_name]['codec']}", "wb")
            )

        # 循环读取，显示并保存视频流
        while not device.isClosed():
            frame_list = []
            for cam_name in cam_list:
                packet = output_queues[cam_name].tryGet()
                if packet is not None:
                    # 输出视频帧的时间戳
                    print(cam_name + ":", packet.getTimestampDevice())
                    # 获取视频帧并添加到帧列表中
                    frame_list.append((cam_name, packet.getCvFrame()))

                if video_queues[cam_name].has():
                    # 将视频流编码并保存到文件中
                    video_queues[cam_name].get().getData().tofile(video_files[cam_name])

            if frame_list:
                print("-------------------------------")
                # 显示视频帧
                for cam_name, frame in frame_list:
                    cv2.imshow(cam_name, frame)

            # 等待用户按下 "q" 键，退出循环并关闭窗口
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

        print("要查看编码后的数据，使用下面的命令将流文件（.mjpeg/.ḣ264/.ḣ265）转换成视频文件（.mp4）:")
        for cam_name in cam_list:
            codec = cam_list[cam_name]["codec"]
            print(f"ffmpeg -i {cam_name}.{codec} -c copy {cam_name}.mp4")


if __name__ == "__main__":
    main()
