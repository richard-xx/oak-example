[oak-ffc-4p](oak-ffc-4p.py)

> 利用 DepthAI 库捕获多个摄像头的视频流，
> 使用 OpenCV 实时显示它们，并且用户可以通过按下“q”键退出程序。
> 脚本定义了一个摄像头字典，包括它们的属性，如颜色或分辨率，
> 并为每个摄像头在 DepthAI 管道中创建了一个节点。
> 然后，该管道被执行在 DepthAI 设备上，以从每个摄像头捕获和检索帧，
> 并使用 OpenCV 在单独的窗口中显示。
> 这个项目可以作为开发多摄像头应用程序（例如监控摄像头或机器人应用程序）的起点，
> 具有很高地可扩展性和可定制性。


[oak-ffc-save-png.py](oak-ffc-save-png.py)

> 根据 `oak-ffc-4p` 修改 添加用户可以通过按下“c”键保存图像。

[oak-ffc-4p-sync-ar0234](oak-ffc-4p-sync-ar0234.py)

> 根据 `oak-ffc-4p` 修改 实现 ar0234 相机同步

[oak-ffc-4p-sync-ov9782](oak-ffc-4p-sync-ov9782.py)

> 根据 `oak-ffc-4p` 修改
> 
> 实现了一个多摄像头实时采集和显示的功能，并实现了按下 
> `c`
> 键进行截图保存的功能。
> 
> 当按下
> `c`
> 键时，会保存每个摄像头最新的一帧图像为
> `PNG`
> 格式。
> 
> 再次按下
> `c`
> 键时，会停止保存。
> 

[oak-ffc-4p-sync-ov9282](oak-ffc-4p-sync-ov9282.py)

> 根据 `oak-ffc-4p` 修改 实现 ov9282 相机同步