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

with dai.Device(pipeline) as device:
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
        q[cam_name] = device.getOutputQueue(name=cam_name, maxSize=1, blocking=False)
        cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam_name, 640, 480)

    while True:
        frame_list = []
        for cam_name in cam_list:
            pkt = q[cam_name].tryGet()
            if pkt is not None:
                frame = pkt.getCvFrame()
                frame_list.append((cam_name, frame))

        for cam_name, frame in frame_list:
            cv2.imshow(cam_name, frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()