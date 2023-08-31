# coding=utf-8
from __future__ import annotations

import time

import cv2
import depthai as dai

cam_list = {
    "CAM_A": {"color": True, "res": "1200"},
    "CAM_B": {"color": True, "res": "1200"},
    "CAM_C": {"color": True, "res": "1200"},
    "CAM_D": {"color": True, "res": "1200"},
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
        cam[cam_name].initialControl.setExternalTrigger(4, 3)

    script = pipeline.create(dai.node.Script)
    script.setScript(
        """# coding=utf-8
import GPIO
import time
GPIO_PIN=41 # Trigger

GPIO.setup(GPIO_PIN, GPIO.OUT, GPIO.PULL_DOWN)

def capture():
    GPIO.write(GPIO_PIN, True)
    time.sleep(0.001) # 1ms pulse is enough
    GPIO.write(GPIO_PIN, False)

while True:
    time.sleep(0.5) 
    capture()
    node.warn('Trigger successful')
"""
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
                print("-------------------------------")
                # 显示视频帧
                for cam_name, frame in frame_list:
                    cv2.imshow(cam_name, frame)

            # 等待用户按下 "q" 键，退出循环并关闭窗口
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
