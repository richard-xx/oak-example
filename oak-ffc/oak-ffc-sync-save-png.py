# coding=utf-8
from datetime import datetime
from multiprocessing import JoinableQueue, Process
from pathlib import Path
from queue import Empty, Full

import cv2
import depthai as dai

from utils import PairingSystem

dest = Path(__file__).parent.joinpath("data")
dest.mkdir(parents=True, exist_ok=True)

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
        # cam[cam_name].initialControl.setExternalTrigger(4, 3)
        # cam[cam_name].setFps(20.0)

        if cam_name == "CAM_A":
            cam[cam_name].initialControl.setFrameSyncMode(
                dai.CameraControl.FrameSyncMode.OUTPUT
            )
        else:
            cam[cam_name].initialControl.setFrameSyncMode(
                dai.CameraControl.FrameSyncMode.INPUT
            )

    return pipeline


def store_frames(queue1: JoinableQueue, queue2: JoinableQueue):
    global sensor_names
    current_queue = queue1

    while True:
        try:
            frames_dict = current_queue.get()
        except Empty:
            current_queue = queue2 if current_queue is queue1 else queue1
            continue

        capture_time = datetime.now().strftime("%Y%m%d_%H%M%S.%f")

        for cam_name, frame in frames_dict.items():
            width, height = frame.shape[:2]

            capture_file_name = (
                f"{dest}/capture_{cam_name}_{sensor_names[cam_name]}"
                f"_{width}x{height}_"
                f"{capture_time}.png"
            )

            print("Saving:", capture_file_name)
            # 将图像编码为 PNG 格式并写入文件
            ruf, image_data = cv2.imencode(".png", frame)
            if ruf:
                image_data.tofile(capture_file_name)

        current_queue.task_done()


def main():
    global cam_list, sensor_names

    # 创建 DepthAI 设备配置对象
    config = dai.Device.Config()

    # 设置 GPIO 引脚 6 为输出模式，初始状态为高电平
    config.board.gpio[6] = dai.BoardConfig.GPIO(
        dai.BoardConfig.GPIO.OUTPUT, dai.BoardConfig.GPIO.Level.HIGH
    )
    # 设置 OpenVINO 版本号
    config.version = dai.OpenVINO.VERSION_2021_4

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

        # 创建相机输出队列
        output_queues = {}
        for cam_name in cam_list:
            output_queues[cam_name] = device.getOutputQueue(
                name=cam_name, maxSize=4, blocking=False
            )
            cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam_name, 640, 480)

        # 创建 PairingSystem 对象，用于匹配不同相机的数据包
        ps = PairingSystem()
        ps.ts_streams = list(cam_list.keys())[:1]
        ps.seq_streams = list(cam_list.keys())[1:]
        ps.threshold = 0.0003  # ms

        # 创建多进程，用于将保存的帧数据写入磁盘
        save = False
        frame_q1 = JoinableQueue(30)
        frame_q2 = JoinableQueue(30)
        current_queue = frame_q1

        store_p = Process(target=store_frames, args=(frame_q1, frame_q2), daemon=True)
        store_p.start()

        # 循环处理视频流数据
        while not device.isClosed():
            # 从相机输出队列中获取数据包，并通过 PairingSystem 对象将不同相机的数据包匹配成一组
            for cam_name in cam_list:
                packets = output_queues[cam_name].tryGetAll()
                ps.add_packets(packets, cam_name)

            pairs = ps.get_pairs()
            for pair in pairs:

                extracted_pair = {}

                print("-------------------------------")
                for cam_name, packet in pair.items():
                    print(cam_name + ":", packet.getTimestampDevice())
                    # 将数据包转换成 OpenCV 格式的帧，并显示在屏幕上
                    frame = packet.getCvFrame()
                    cv2.imshow(cam_name, frame)
                    extracted_pair.update({cam_name: frame})

                # 如果开启了保存帧数据的功能，则将匹配的帧数据存入队列中
                if save:
                    try:
                        current_queue.put(extracted_pair)
                    except Full:
                        current_queue = (
                            frame_q2 if current_queue is frame_q1 else frame_q1
                        )

                        current_queue.put(extracted_pair)

            # 处理按键事件
            key = cv2.waitKey(1)
            if key == ord("output_queues"):
                break
            elif key == ord("c"):
                save = not save
                if save:
                    print("开始保存图片")
                else:
                    print("暂停保存图片")

        # 关闭 OpenCV 窗口，并等待独立进程结束
        cv2.destroyAllWindows()
        frame_q1.join()
        frame_q2.join()


if __name__ == "__main__":
    main()
