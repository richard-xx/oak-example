# coding=utf-8
from __future__ import annotations

import cv2
import depthai as dai
import numpy as np

cam_list = {
    "stereo_ad": {
        "left": {"socket": "CAM_A", "color": True, "res": "800"},
        "right": {"socket": "CAM_D", "color": True, "res": "800"},
    },
    "stereo_bc": {
        "right": {"socket": "CAM_C", "color": True, "res": "800"},
        "left": {"socket": "CAM_B", "color": True, "res": "800"},
    },
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
    stereo = {}
    streams = []

    def create_cam(cam_name, stereo_name, cam_props, position):
        streams.append(cam_name)
        xout[cam_name] = pipeline.create(dai.node.XLinkOut)
        xout[cam_name].setStreamName(cam_name)
        if cam_props["color"]:
            cam[cam_name] = pipeline.create(dai.node.ColorCamera)
            cam[cam_name].setResolution(color_res_opts[cam_props["res"]])
            if cam_props["res"] == "1200":
                cam[cam_name].setIspScale(1, 3)
        else:
            cam[cam_name] = pipeline.create(dai.node.MonoCamera)
            cam[cam_name].setResolution(mono_res_opts[cam_props["res"]])
        cam[cam_name].setBoardSocket(cam_socket_opts[cam_name])
        link_cam(cam_name, stereo[stereo_name], position)

    def link_cam(cam_name, stereo_, position):
        (link_to_stereo_left if position == "left" else link_to_stereo_right)(
            cam[cam_name], stereo_, xout[cam_name].input,
        )

    def link_to_stereo_left(cam_, stereo_input, xout_input):
        # (cam_.isp if hasattr(cam, "isp") else cam_.out).link(stereo_input.left)
        if hasattr(cam_, "isp"):
            cam_.isp.link(stereo_input.left)
        else:
            cam_.out.link(stereo_input.left)
        stereo_input.syncedLeft.link(xout_input)

    def link_to_stereo_right(cam_, stereo_input, xout_input):
        if hasattr(cam_, "isp"):
            cam_.isp.link(stereo_input.right)
        else:
            cam_.out.link(stereo_input.right)
        stereo_input.syncedRight.link(xout_input)

    for stereo_name, cam_group in cam_list.items():
        stereo[stereo_name] = pipeline.create(dai.node.StereoDepth)
        stereo[stereo_name].setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo[stereo_name].setLeftRightCheck(enable=True)
        stereo[stereo_name].setSubpixel(enable=False)
        stereo[stereo_name].setExtendedDisparity(enable=False)
        stereo[stereo_name].initialConfig.setMedianFilter(
            dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
        )
        xout[stereo_name] = pipeline.create(dai.node.XLinkOut)
        xout[stereo_name].setStreamName(stereo_name)
        streams.append(stereo_name)
        stereo[stereo_name].depth.link(xout[stereo_name].input)
        for position, cam_props in cam_group.items():
            cam_name = cam_props["socket"]
            if cam_name not in streams:
                create_cam(cam_name, stereo_name, cam_props, position)
                cam[cam_name].setBoardSocket(cam_socket_opts[cam_name])
            else:
                link_cam(cam_name, stereo[stereo_name], position)

    return pipeline, streams


def main():
    # 创建设备对象，并获取连接的相机特性信息
    with dai.Device() as device:
        print("Connected cameras:")
        sensor_names = {}
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

        # # 仅保留设备已连接的相机
        # for cam_name in set(cam_list).difference(sensor_names):
        #     print(f"{cam_name} is not connected !")

        # cam_list = {name: cam_list[name] for name in set(cam_list).intersection(sensor_names)}

        # 开始执行给定的管道
        pipeline, streams = create_pipeline()
        device.startPipeline(pipeline)

        # 创建输出队列和显示窗口
        output_queues = {}
        for cam_name in streams:
            output_queues[cam_name] = device.getOutputQueue(
                name=cam_name,
                maxSize=1,
                blocking=False,
            )
            cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam_name, 640, 480)

        # 循环读取并显示视频流
        while not device.isClosed():
            frame_list = []
            for cam_name in streams:
                packet = output_queues[cam_name].get()
                if packet is not None:
                    # 输出视频帧的时间戳
                    # print(cam_name + ":", packet.getTimestampDevice())
                    # 获取视频帧并添加到帧列表中
                    frame_list.append((cam_name, packet.getCvFrame()))

            # print("-------------------------------")
            # 显示视频帧
            for cam_name, frame in frame_list:
                if cam_name in cam_list:
                    depth_downscaled = frame[::4]
                    non_zero_depth = depth_downscaled[depth_downscaled != 0]  # Remove invalid depth values
                    if len(non_zero_depth) == 0:
                        min_depth, max_depth = 0, 0
                    else:
                        min_depth = np.percentile(non_zero_depth, 3)
                        max_depth = np.percentile(non_zero_depth, 97)
                    depth_colorized = np.interp(frame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
                    depth_colorized = cv2.applyColorMap(depth_colorized, cv2.COLORMAP_JET)

                    cv2.imshow(cam_name, depth_colorized)
                else:
                    cv2.imshow(cam_name, frame)

            # 等待用户按下 "q" 键，退出循环并关闭窗口
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
