# 2019 短期实验归档

成绩见 [RESULTS_2019.md](../../RESULTS_2019.md)。窗口为 2019-01-02 00–07 时 GPS，每小时一次，全部保留。

- `cpu/`、`gpu/`：原始 TimesFM 的实际运行结果；逐点坐标、历史误差、指标、图表、环境与源码哈希。
- `dynamics-cache/`：16 条由历史位置拟合、积分的动力学轨道，不包含未来参考坐标。
- `source/`：运行时源码字节副本，与环境记录的 SHA-256 一致。
- `validation/integration.json`：第 3 窗口更严格积分设置的敏感度检查。
- `analysis.json`：逐点评分复核、改善窗口数、CPU/GPU 最大差异及权重 SHA-256。

运行 `python short_2019.py --device cpu` 或 `--device gpu` 重做 TimesFM 预报。
加 `--model physics --refit --workers 4` 可从头拟合动力学。数据哈希、力模型、源码或依赖变化会使缓存失效。
运行时写入 `outputs/`；这个归档不覆盖。

数据源为 GFZ 镜像的 JPL GNV1B RL04，源地址及哈希见 [数据清单](../../data/orbit-2019.sources.json)。
运行 `python advanced/download_2019.py` 可重建附带的 10 秒采样 CSV，下载约 317 MB。
TimesFM 原始权重 revision 为 `43046b85ec22d584a13f8098c2ed39c889e129c2`，未微调。

CPU/GPU 的环境记录运行时间包括缓存读取与实际模型推理，不包含提前计算动力学缓存的耗时。
本次 CPU 以 8 个工作进程计算 16 条动力学轨道。结果不是论文全年测试集的复现。
