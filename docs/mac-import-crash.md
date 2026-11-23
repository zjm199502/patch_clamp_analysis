# M4 / M5 加载记录后闪退

2026-11-23 登记。状态为暂缓排查，等待在发生闪退的 Mac 上再次观察。

用户反馈，M4 和 M5 Mac 上的 ClampTrace 可以启动，加载记录后应用退出。发生故障的具体软件版本、macOS 版本、导入入口、文件格式、记录规模和崩溃栈尚待确认。

已用发布的 Apple Silicon 安装包和 0.3.3 补丁在 M1 虚拟机的原生 Cocoa 窗口测试。合成数据包括一个 10 个 sweep、共 50 万个采样点的 ABF 和一个 400 万个采样点的 HDF5。两份文件均完成导入、预览、保存及项目重开。本次未复现用户报告的退出，原因待查。[测试记录](https://github.com/zjm199502/patch_clamp_analysis/actions/runs/45095483035)

下次复现时，先记录闪退时间、软件和 macOS 版本，以及从选择文件到退出的最后一步。如果「控制台 → 崩溃报告」有同一时间的 PatchClampAnalysis 报告，可附上对应的 `.ips`，或提供 Exception Type、Termination Reason 和崩溃线程的调用栈。
