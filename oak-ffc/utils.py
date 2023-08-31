# coding=utf-8
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import depthai as dai


def step_norm(value, threshold=0.0003):
    """
    将时间戳标准化，将其舍入到最接近的 threshold 的倍数

    Args:
        value (float): 时间戳值
        threshold (float, optional): 用于舍入的阈值。Defaults to 0.3.

    Returns:
        float: 标准化后的时间戳值
    """
    return round(value / threshold) * threshold


def seq(packet):
    """
    返回数据包的序列号

    Args:
        packet: 数据包

    Returns:
        int: 序列号
    """
    return packet.getSequenceNum()


def tst(packet):
    """
    返回数据包的时间戳（单位为秒）

    Args:
        packet: 数据包

    Returns:
        float: 时间戳（单位为秒）
    """
    return packet.getTimestamp().total_seconds()


def has_keys(obj, keys):
    """
    检查一个对象是否包含指定的键

    Args:
        obj: 对象
        keys (list): 键列表

    Returns:
        bool: 如果对象包含所有指定键，则返回 True；否则返回 False
    """
    return all(stream in obj for stream in keys)


class PairingSystem:
    seq_streams = ["CAM_B", "CAM_C", "CAM_D"]
    ts_streams = ["CAM_A"]
    seq_ts_mapping_stream = "CAM_B"

    threshold = 0.3  # 用于舍入的阈值

    def __init__(self):
        """初始化 PairingSystem 类的实例。"""
        # 初始化类的属性
        self.ts_packets = {}
        self.seq_packets = {}
        self.last_paired_ts = None
        self.last_paired_seq = None

    def add_packets(
        self,
        packets: dai.ImgFrame | list[dai.ImgFrame] | None,
        stream_name: str,
    ) -> None:
        """
        将数据包添加到该类的数据结构中，数据包来自于指定的流。

        Args:
            packets (Union[dai.ImgFrame, List[dai.ImgFrame]]): 数据包。
            stream_name (str): 数据包所属的流的名称。
        """
        if packets is None:
            return
        # 如果数据包来自序列号流，则将其添加到序列号数据包字典中
        if stream_name in self.seq_streams:
            for packet in packets:
                seq_key = seq(packet)
                self.seq_packets[seq_key] = {
                    **self.seq_packets.get(seq_key, {}),
                    stream_name: packet,
                }
        # 如果数据包来自时间戳流，则将其添加到时间戳数据包字典中
        elif stream_name in self.ts_streams:
            for packet in packets:
                ts_key = step_norm(tst(packet), self.threshold)
                self.ts_packets[ts_key] = {
                    **self.ts_packets.get(ts_key, {}),
                    stream_name: packet,
                }

    def get_pairs(self) -> list[dict[str, Any]]:
        """
        在已添加的数据包中查找匹配的时间戳和序列号，并返回匹配的数据包。

        Returns:
            List[Dict[str, Any]]: 匹配的数据包列表。
        """
        results = []
        # 遍历所有已添加的序列号数据包
        for key in list(self.seq_packets.keys()):
            # 如果数据包中包含了所有序列号流的数据，则进行匹配
            if has_keys(self.seq_packets[key], self.seq_streams):
                # 计算序列号对应的时间戳
                ts_key = step_norm(
                    tst(self.seq_packets[key][self.seq_ts_mapping_stream]),
                    self.threshold,
                )
                # 如果时间戳数据包中包含了所有时间戳流的数据，则进行匹配
                if ts_key in self.ts_packets and has_keys(
                    self.ts_packets[ts_key], self.ts_streams
                ):
                    # 将匹配的数据包添加到结果列表中
                    results.append({**self.seq_packets[key], **self.ts_packets[ts_key]})
                    # 记录上一个匹配的序列号和时间戳
                    self.last_paired_seq = key
                    self.last_paired_ts = ts_key
        # 如果有匹配的数据包，则删除已匹配的数据包，以便节省内存空间
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self) -> None:
        """删除已匹配的数据包，以便节省内存空间。"""
        # 删除所有序列号小于等于上一个匹配的序列号的数据包
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]
        # 删除所有时间戳小于等于上一个匹配的时间戳的数据包
        for key in list(self.ts_packets.keys()):
            if key <= self.last_paired_ts:
                del self.ts_packets[key]
