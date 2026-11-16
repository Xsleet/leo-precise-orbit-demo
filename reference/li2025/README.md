# Li（2025）思路：CPU/GPU 实测归档

默认 EGM2008 120 阶/次动力学，八个固定窗口均运行，没有挑选窗口。

- cpu/、gpu/：每窗历史输入、未来预测、图、拟合参数及总体指标。
- environment.json：每次批处理的设备、库版本、源码哈希与耗时。
- source/：本次计算时的源码快照；日常运行请使用仓库根目录（含 data/）。
- j2-gpu/、j2-source/：加入高阶重力场前的完整 J2 轮次及源码快照。
- analysis.json：相对改善比例与改善窗口数。
- gravity-validation.json：独立引力实现核对。
- integration-validation.json：积分容差敏感性诊断。

可读结果见 [RESULTS.md](../../RESULTS.md)。
运行命令为 python compare_windows.py --device cpu 或 --device gpu。
新输出进入 outputs/，不会覆盖本目录。

本次 CPU/GPU 均设置 OMP_NUM_THREADS=4、MKL_NUM_THREADS=4。
动力学都在 CPU；GPU 为 RTX 5070 Ti。CPU/GPU 结果允许浮点差异。
CI 重算基础动力学、边界和归档指标，不在云端重新下载运行 TimesFM。
原始 TimesFM 3.0 权重固定到 43046b85ec22d584a13f8098c2ed39c889e129c2，无微调。

运行后仅更正了根目录 rdop.py 顶部的旧 J2 说明文字，计算代码未改变；source/ 保留运行时原文件，用于核对 environment.json 中的哈希。
