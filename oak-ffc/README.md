[oak-ffc](oak-ffc.py)

> 利用 DepthAI 库捕获多个摄像头的视频流，
> 使用 OpenCV 实时显示它们，并且用户可以通过按下 `q` 键退出程序。
> 脚本定义了一个摄像头字典，包括它们的属性，如颜色或分辨率，
> 并为每个摄像头在 DepthAI 管道中创建了一个节点。
> 然后，该管道被执行在 DepthAI 设备上，以从每个摄像头捕获和检索帧，
> 并使用 OpenCV 在单独的窗口中显示。
> 这个项目可以作为开发多摄像头应用程序（例如监控摄像头或机器人应用程序）的起点，
> 具有很高地可扩展性和可定制性。

[oak-ffc-sync-ar0234](oak-ffc-sync-ar0234.py)

> 根据 `oak-ffc` 修改 实现 ar0234 相机同步

[oak-ffc-sync-ov9*82](oak-ffc-sync-ov9*82.py)

> 根据 `oak-ffc` 修改 实现 ov9*82 相机同步

[oak-ffc-save-png](oak-ffc-save-png.py)

> 根据 `oak-ffc` 修改 
> 
> 它会在不同的窗口中显示视频流，并允许用户通过按下 `c` 键来捕获视频帧。
> 
> 捕获的帧将以 PNG 格式保存到磁盘上，文件名包括相机名称、传感器名称、分辨率、曝光时间、ISO、镜头位置、捕获时间和帧序号。


[oak-ffc-sync-save-png](oak-ffc-sync-save-png.py)

> 根据 `oak-ffc` 修改
> 
> 这是一个使用 DepthAI SDK 和 OpenCV 库实现的多相机视频流处理程序，
> 读取并显示相机输出流的图像，
> 并将相机输出流保存为 PNG 图像文件。
> 
> 程序使用了多进程技术，通过队列将帧数据传递给独立的进程来保存图像文件。
> 
> 程序还实现了相机输出流的帧同步功能，确保不同相机捕获到的帧时刻相同。
> 
> 程序使用了 PairingSystem 类来匹配不同相机输出流之间的帧数据，
> 确保输出流之间的帧数据是匹配的。

[oak-ffc-sync-video-encoding](oak-ffc-sync-video-encoding.py)

> 根据 `oak-ffc` 修改 
> 
> 连接多个相机，并将它们的视频流编码为 H.265 或 H.264，然后保存为文件，并在窗口中实时显示视频流。
> 
> 要查看编码后的数据，使用下面的命令将流文件（.mjpeg/.ḣ264/.ḣ265）转换成视频文件（.mp4）：
> 
> `ffmpeg -i CAM_A.h265 -c copy CAM_A.mp4`

[oak-ffc-depth](oak-ffc-depth.py)

> 多组深度
