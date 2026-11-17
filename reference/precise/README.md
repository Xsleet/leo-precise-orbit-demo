# 精密动力学 + TimesFM：完整复现记录

本目录保留 2025 实验。其成绩见 [RESULTS.md](../../RESULTS.md)；新 2019 短期实验见 [../2019-short](../2019-short/README.md)。
本目录的缓存对应 `source/` 中的旧源码。当前版本修复了跨年份天气读取并支持 10 s 拟合，因此旧源码缓存会正常失效；重跑 2025 可重新拟合，或从 Git 提交 `78dead04bb04f51200240e3bc8866b4255ddcbaf` 恢复原版完整环境。

- `cpu/`、`gpu/`：原始 TimesFM 分别在 CPU、RTX 5070 Ti 上实际推理的八个固定窗口，含逐点坐标、历史误差、图表、汇总和论文对照。
- `dynamics-cache/`：用过去数据实际拟合、积分得到的 16 条动力学轨道；每个窗口含旧弧段和最新历史重拟合各一条。没有未来参考轨道列。
- `source/`：运行时源码的原始字节副本，与 `environment.json` 的 SHA-256 对应。
- `validation/`：积分收敛、变分导数与独立扰动积分的比较。
- `analysis.json`：改善窗口数、CPU/GPU 最大差异和已校验权重哈希。
- `prefit_timing.json`：实际动力学拟合的耗时记录。本次开发机同时运行 8 个独立 CPU 进程。

CPU/GPU 共用同一套 CPU 动力学结果，随后分别运行 TimesFM。`environment.json` 的运行时间包含缓存读取和实际 TimesFM 推理，不包含此前的动力学拟合时间。

## 运行

```bash
python compare_windows.py --device cpu
python compare_windows.py --device gpu
```

默认复用与源码、历史、力模型、依赖版本匹配的缓存。需要从头检验拟合过程时：

```bash
python main.py --model physics --window 1 --refit
python compare_windows.py --device cpu --refit
```

重新运行写入 `outputs/`，本归档保持不变。首次安装可能需要下载模型权重；已有对应模型目录时可加 `--checkpoint 路径`。

模型源码提交 `e51928e27119cb17bebc005be2696b75e0a9e688`，权重 revision `43046b85ec22d584a13f8098c2ed39c889e129c2`。
权重 SHA-256 为 `a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`，未微调。

## 核验

`python -m unittest discover -v` 检查：

1. 力模型文件哈希、150 阶配置、GPS/UTC 时标与精密坐标往返。
2. EOT11a 原始 GFC 系数与 Orekit 实际分潮计算的一致性。
3. 修改未来空间天气数据不会改变预报输入；最后一段阻力尺度具有完整周期的历史观测。
4. 变分方程导数与独立扰动积分相符。
5. 从逐点坐标重算所有窗口的成绩，按拟合参数重新积分恢复归档轨道。

CI 不下载或重新运行 TimesFM 权重，云端检查通过不代表重新进行了一次大模型推理。真实推理环境记录在 CPU/GPU 目录中。

上一轮简化力模型保存在 [../li2025](../li2025/README.md)，不能混入本轮统计。
