# coding=utf-8
from __future__ import annotations

import math
from collections import OrderedDict
from datetime import timedelta
from typing import Any, Callable, overload

import cv2  # type: ignore
import depthai as dai  # type: ignore
import numpy as np

COlOR_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1080_P
MONO_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P
DOWNSCALE_COLOR = True
BLOCKING = False
FPS = 30
SYNC_BASIS = "sequence"  # timestamp, sequence


def create_pipeline(device):
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(COlOR_RESOLUTION)
    if DOWNSCALE_COLOR:
        cam_rgb.setIspScale(2, 3)  # 1080P -> 720P
    cam_rgb.setFps(FPS)

    try:
        calib_data = device.readCalibration2()
        lens_position = calib_data.getLensPosition(dai.CameraBoardSocket.CAM_A)
        if lens_position:
            cam_rgb.initialControl.setManualFocus(lens_position)
    except:
        raise

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")

    cam_rgb.isp.link(xout_rgb.input)

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(MONO_RESOLUTION)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setFps(FPS)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(MONO_RESOLUTION)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setFps(FPS)

    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    mono_left.out.link(xout_left.input)

    xout_right = pipeline.create(dai.node.XLinkOut)
    xout_right.setStreamName("right")
    mono_right.out.link(xout_right.input)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.initialConfig.setLeftRightCheck(True)
    stereo.initialConfig.setExtendedDisparity(False)
    stereo.initialConfig.setSubpixel(False)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    xout_disparity = pipeline.create(dai.node.XLinkOut)
    xout_disparity.setStreamName("disparity")
    stereo.disparity.link(xout_disparity.input)

    return (
        pipeline,
        stereo.initialConfig.getMaxDisparity(),
        (
            stereo.initialConfig.get().postProcessing.thresholdFilter.minRange,
            stereo.initialConfig.get().postProcessing.thresholdFilter.maxRange,
        ),
    )


class SortedDict(OrderedDict):
    """A dictionary that maintains the keys in a sorted order based on a custom sort function.

    Args:
        key_sort_fn (Callable[[Any], Any] | None): A custom sorting function for keys.
    """

    @overload
    def __init__(self, *args, key_sort_fn=None) -> None:
        ...

    @overload
    def __init__(self, *args, reverse=None) -> None:
        ...

    @overload
    def __init__(self, *args, key_sort_fn: Callable[[Any], Any] | None, reverse: bool):
        ...

    def __init__(self, *args, **kwargs) -> None:
        self.key_sort_fn = kwargs.get("key_sort_fn", lambda x: x)
        self.reverse = kwargs.get("reverse", False)
        kwargs = {k: v for k, v in kwargs.items() if k not in ("key_sort_fn", "reverse")}
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self._sort_keys()

    def _sort_keys(self) -> None:
        """Sorts the dictionary keys based on the specified sort function."""
        if self.key_sort_fn:
            sorted_keys = sorted(self.keys(), key=self.key_sort_fn, reverse=self.reverse)
            for key in sorted_keys:
                self.move_to_end(key=key, last=True)
            # ordered_dict = OrderedDict()
            # for key in sorted_keys:
            #     ordered_dict[key] = self[key]
            # self.clear()
            # self.update(ordered_dict)


class MessageSeqSync:
    """A class to manage messages and synchronize them based on sequence numbers."""

    def __init__(self, length=2):
        """Initialize the MessageSync."""
        self.msgs = SortedDict()
        self.length = length

    def add_msg(self, msg, name, seq=None):
        """Add a message to the manager.

        Args:
            msg: The message to add.
            name: The name of the message.
            seq: The sequence number of the message. If not provided, it will be extracted from the message.

        """
        if seq is None:
            seq = msg.getSequenceNum()
        seq = str(seq)
        if seq not in self.msgs:
            self.msgs[seq] = {}
        self.msgs[seq][name] = msg

    def get_sync_msgs(self):
        """Get synchronized messages.

        Returns:
            A dictionary containing synchronized messages, or None if no synchronized messages are found.

        """
        seq_remove = []  # Arr of sequence numbers to get deleted
        for seq, syncMsgs in self.msgs.items():
            seq_remove.append(seq)  # Will get removed from dict if we find synced msgs pair
            # Check if we have both detections and color frame with this sequence number
            if len(syncMsgs) == self.length:  # rgb + depth
                for rm in seq_remove:
                    del self.msgs[rm]
                return syncMsgs  # Returned synced msgs
        return None


class MessageTimeSync:
    """A class to manage messages and synchronize them based on sequence numbers."""

    def __init__(self, length=2, fps=30):
        """Initialize the MessageManager."""
        self.msgs = {}
        self.sync_msgs = SortedDict()  # SortedDict to keep track of sync messages

        self._length = length
        self._fps = fps
        self._thresh = timedelta(milliseconds=math.ceil(500 / self._fps))
        print(f"MessageTimeSync: Threshold is {self._thresh.microseconds / 1000} ms")

    def add_msg(self, msg, name, timestamp=None):
        """Add a message to the manager.

        Args:
            msg: The message to add.
            name: The name of the message.
            timestamp: The sequence number of the message. If not provided, it will be extracted from the message.

        """
        if timestamp is None:
            timestamp = msg.getTimestamp()

        if name not in self.msgs:
            self.msgs[name] = SortedDict()

        sync_flag = False
        sync_msgs_copy = list(self.sync_msgs.items())
        for i, (matching_frames, min_timestamp, max_timestamp) in sync_msgs_copy:
            if name not in matching_frames and (
                abs(timestamp - min_timestamp) < self._thresh or abs(timestamp - max_timestamp) < self._thresh
            ):
                matching_frames[name] = msg
                min_timestamp = min(min_timestamp, timestamp)
                max_timestamp = max(max_timestamp, timestamp)
                self.sync_msgs.pop(i)
                self.sync_msgs[min_timestamp] = (
                    matching_frames,
                    min_timestamp,
                    max_timestamp,
                )
                sync_flag = True
                break

        if not sync_flag:
            matching_frames = {name: msg}
            min_timestamp = timestamp
            max_timestamp = timestamp

            for other_name, other_msgs in self.msgs.items():
                if other_name != name:
                    for other_timestamp in other_msgs:
                        time_diff = abs(timestamp - other_timestamp)
                        if time_diff <= self._thresh:
                            min_timestamp = min(min_timestamp, other_timestamp)
                            max_timestamp = max(max_timestamp, other_timestamp)
                            matching_frames[other_name] = other_msgs[other_timestamp]
                            del self.msgs[other_name][other_timestamp]
                            sync_flag = True
                            break

            if len(matching_frames) > 1:
                self.sync_msgs[min_timestamp] = (
                    matching_frames,
                    min_timestamp,
                    max_timestamp,
                )
        if not sync_flag:
            self.msgs[name][timestamp] = msg

    def get_sync_msgs(self):
        """Get synchronized messages.

        Returns:
            A dictionary containing synchronized messages, or None if no synchronized messages are found.

        """
        seq_remove = []
        sync_msgs_copy = self.sync_msgs.copy()
        for timestamp, (matching_frames, _min_timestamp, _max_timestamp) in sync_msgs_copy.items():
            seq_remove.append(timestamp)
            if len(matching_frames) == self._length:
                for rm in seq_remove:
                    self.sync_msgs.pop(rm)
                return matching_frames
        return None

    @property
    def thresh(self):
        return self._thresh

    @thresh.setter
    def thresh(self, value):
        self._thresh = value

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = value
        self._thresh = timedelta(milliseconds=math.ceil(500 / value))

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value


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


def main():
    with dai.Device() as device:
        print("Starting pipeline...")
        pipeline, max_disparity, threshold_range = create_pipeline(device)
        device.startPipeline(pipeline)

        output_names = device.getOutputQueueNames()

        if SYNC_BASIS == "sequence":
            msg_sync = MessageSeqSync(length=len(output_names))
        else:
            msg_sync = MessageTimeSync(length=len(output_names), fps=FPS)

        print("Pipeline is running. Press Ctrl+C to terminate...")
        print("name: SequenceNum, Timestamp")
        while True:
            for output_name in output_names:
                msg = device.getOutputQueue(name=output_name, maxSize=4, blocking=BLOCKING).get()
                if msg is not None:
                    msg_sync.add_msg(msg, output_name)

            sync_msgs = msg_sync.get_sync_msgs()
            if sync_msgs is not None:
                print("", end="\r")
                for name, msg in sync_msgs.items():
                    print(f"{name}: {msg.getSequenceNum()}, {msg.getTimestamp()}", end="; ")

                for name, msg in sync_msgs.items():
                    frame = msg.getCvFrame()
                    if name == "depth":
                        depth_frame = np.interp(frame, threshold_range, (0, 255)).astype(np.uint8)
                        depth_colorized = cv2.applyColorMap(depth_frame, cv2.COLORMAP_TURBO)
                        frame = np.ascontiguousarray(depth_colorized)
                    elif name == "disparity":
                        # disparity_frame = cv2.convertScaleAbs(frame, alpha=(max_disparity / 255))
                        disparity_frame = np.interp(frame, (0, max_disparity), (0, 255)).astype(np.uint8)
                        disparity_colorized = cv2.applyColorMap(disparity_frame, cv2.COLORMAP_TURBO)
                        frame = np.ascontiguousarray(disparity_colorized)

                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                    draw_text(frame, f"SequenceNum: {msg.getSequenceNum()}", (20, 20), (0, 0, 255))
                    draw_text(frame, f"Timestamp: {msg.getTimestamp()}", (20, 40), (0, 0, 255))

                    cv2.imshow(name, frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
