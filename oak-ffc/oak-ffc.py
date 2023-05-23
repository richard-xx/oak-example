# coding=utf-8
import cv2
import depthai as dai


cam_list = {
    "CAM_A": {"color": True, "res": "1080"},
    "CAM_B": {"color": False, "res": "800"},
    "CAM_C": {"color": False, "res": "800"},
    "CAM_D": {"color": True, "res": "1080"},
    # "CAM_E": {"color": False, "res": "1200"},
    # "CAM_F": {"color": False, "res": "1200"},
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
    "CAM_E": "CAM_E",
    "CAM_F": "CAM_F",
}

cam_socket_opts = {
    "CAM_A": dai.CameraBoardSocket.RGB,  # Or CAM_A
    "CAM_B": dai.CameraBoardSocket.LEFT,  # Or CAM_B
    "CAM_C": dai.CameraBoardSocket.RIGHT,  # Or CAM_C
    "CAM_D": dai.CameraBoardSocket.CAM_D,
    "CAM_E": dai.CameraBoardSocket.CAM_E,
    "CAM_F": dai.CameraBoardSocket.CAM_F,
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

    return pipeline


def main():
    global cam_list
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
            cam_feature = cam_list.get(cam_name)
            if cam_feature:
                color_type = "COLOR" if cam_feature.get("color") else "MONO"
                if color_type not in supported_types:
                    cam_feature["color"] = not cam_feature["color"]

            sensor_names[cam_name] = p.sensorName

        # 仅保留设备已连接的相机
        cam_list = {
            name: cam_list[name] for name in cam_list if name in sensor_names.keys()
        }

        # 开始执行给定的管道
        device.startPipeline(create_pipeline())

        # 创建输出队列和显示窗口
        output_queues = {}
        for cam_name in cam_list:
            output_queues[cam_name] = device.getOutputQueue(
                name=cam_name, maxSize=1, blocking=False
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
