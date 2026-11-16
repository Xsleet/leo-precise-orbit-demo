# 低轨卫星精密轨道预报 Demo

用 GRACE-FO C 的真实精密轨道，演示 **动力学预报 + TimesFM 误差预报**。
参考 Li 等（2025）：**过去 8 小时拟合 → 20 分钟预热 → 未来 60 分钟预报**。

当前动力学已升级为 Orekit：150 阶 EGM2008、DE405 日月引力、潮汐、物理阻力、太阳辐射压和精密坐标变换。
TimesFM 使用原始 3.0 权重，没有微调。CPU/GPU 实测成绩及与论文的对照见 [RESULTS.md](RESULTS.md)。

**八个窗口第 60 分钟三维 RMSE：动力学 1.144 m → 加 TimesFM 后 0.761 m。**
动力学基线较旧版的 6.958 m 改善，但整体仍未达到分米级目标。

## 1. 先看懂三个步骤

1. 用过去 8 小时的位置，拟合初始位置/速度、经验加速度和每圈阻力尺度。
2. 保留起报前 20 分钟，看这条动力学预报最近错了多少。
3. **最终预报 = 动力学未来位置 + TimesFM 预测的未来误差**。

TimesFM 输入最近 6 小时的沿轨/法向/径向误差。未来参考轨道只用于最后评分。
先读 [基础知识](BASICS.md)，再读 [main.py](main.py) 的 `predict()`；底层力模型在 [precise.py](precise.py)。
与论文的逐项配置差异见 [DYNAMICS.md](DYNAMICS.md)：大气模型采用 DTM2000 替代 DTM94，仍有名义姿态等简化。

## 2. 下载与运行

安装 Python 3.11 或 3.12 和 Git：

```bash
git clone https://github.com/Xsleet/leo-precise-orbit-demo.git
cd leo-precise-orbit-demo
python -m venv .venv
```

Windows 命令提示符执行 `.venv\Scripts\activate.bat`；Linux/macOS 执行 `source .venv/bin/activate`。
PowerShell 可把下文 `python` 换成 `.\.venv\Scripts\python.exe`。

**CPU：**

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-timesfm.txt
python main.py --device cpu
```

**GPU：NVIDIA 显卡，驱动支持 CUDA 12.8。**

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-timesfm.txt
python main.py --device gpu
```

macOS 的 PyTorch 安装命令改为 `python -m pip install torch==2.8.0`，运行时选 `cpu`。
首次自动下载约 1.32 GB TimesFM 权重；Java 运行时由依赖包安装，无需手动配置。

**默认复用附带的动力学计算结果，再实际运行 TimesFM。** 缓存与历史数据、源码、力模型和依赖版本校验匹配。
研究拟合过程时加 `--refit` 从头计算，耗时明显更长。GPU 只加速 TimesFM，动力学使用 CPU。

不下载 TimesFM，只看动力学和线性误差对照：

```bash
python -m pip install -r requirements.txt
python main.py --model physics
```

## 3. 看懂结果

![过去6小时误差、预热、未来60分钟](data/example.png)

前三行是沿轨、法向、径向误差：**过去 6 小时 → 蓝色预热区 → 起报虚线 → 未来 60 分钟**。
黑色点线是事后参考误差；最下面是各方法的三维位置误差。

保留四个对照：动力学、动力学＋线性误差、动力学＋TimesFM、利用截至起报时最新历史重新拟合的动力学。
后一个对照可以检查：加入 TimesFM 的收益是否超过及时更新动力学本身。

全部八个窗口的整段 RMS、60 min RMSE、逐窗口结果以及论文对照见 **[RESULTS.md](RESULTS.md)**。
“60 min RMSE”指八个窗口在第 60 分钟的误差均方根，与论文 Table 5 的统计口径对应；不同年份、力模型细节和训练规模仍使两者不能视为同场竞赛。

## 4. 自己分析

```bash
python main.py --device cpu --window 3
python compare_windows.py --device cpu
python compare_windows.py --device gpu
python main.py --model physics --refit
```

八个固定窗口为 2025 年 6 月 2–3 日每天 00、06、12、18 时（GPS），全部保留。
已有模型时加 `--checkpoint 模型目录`。旧的简化动力学可用 `--gravity egm2008` 或 `--gravity j2` 重跑。

结果在 `outputs/li2025-precise-timesfm-cpu/` 或对应 GPU 目录：

| 文件 | 用途 |
|---|---|
| window-01/result.png | 历史到未来的图 |
| window-01/history.csv | 历史位置、动力学残差、预热标记 |
| window-01/prediction.csv | 未来参考和各方法预测坐标 |
| window-01/run.json | 拟合参数、时间边界、模型版本 |
| summary.csv / by_window.csv | 整体与逐窗口成绩 |
| by_lead.csv / paper_comparison.csv | 不同预报时长和论文对照 |

当前归档在 [reference/precise](reference/precise/README.md)，旧实验保存在 [reference/li2025](reference/li2025/README.md)。
运行 `python -m unittest discover -v` 检查力模型数据、时标、潮汐相位、历史边界与归档结果。

给学生的三个问题：为什么要预热？动力学更精确后 TimesFM 还能改善多少？为什么换年份和空间天气条件会影响结果？

这是后处理精密产品的离线回放，未模拟产品发布延迟。数据、力模型差异与辅助量可用性见 [DYNAMICS.md](DYNAMICS.md)、[DATA_SOURCES.md](DATA_SOURCES.md)，实验记录见 [EXPERIMENT.md](EXPERIMENT.md)。
自编代码 MIT；数据及模型遵循各自许可。TimesFM 3.0 用于其许可允许的非商业、非生产教学研究，见 [模型许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)。

同系列：[精密钟差预报](https://github.com/Xsleet/gracefo-clock-prediction) · [空间碎片轨道预测](https://github.com/Xsleet/space-debris-orbit-demo)
