# 低轨卫星精密轨道预报 Demo

用 GRACE-FO C 的真实精密轨道，演示 **动力学预报 + TimesFM 误差预报**。
参考 Li 等（2025）的实验结构：**8 小时历史拟合 → 20 分钟预热 → 未来 60 分钟预报**。

**本次 8 个固定窗口：整段三维 RMS 从 5.871 m 降到 0.858 m。**
CPU/GPU 均已实测；使用原始 TimesFM 3.0，没有微调。精密指参考轨道产品，本例尚未达到论文厘米级预报。

## 1. 先看懂三个步骤

1. 用过去 8 小时精密轨道，拟合动力学初始位置、速度及经验加速度。
2. 留出起报前 20 分钟，看这条动力学预报最近错了多少；与前面的拟合残差一起输入 TimesFM。
3. **最终预报 = 动力学未来位置 + TimesFM 预测的未来误差**。

输入 TimesFM 的是最近 6 小时沿轨/法向/径向误差，不直接预测巨大的 XYZ 坐标。
默认动力学为 EGM2008 120 阶/次重力场 + 经验加速度，底层计算与主流程分开。

先读 [基础知识与 10:00 起报例子](BASICS.md)，再读 [main.py](main.py) 的 predict()。
拟合和积分在 [rdop.py](rdop.py)，高阶引力在 [gravity.py](gravity.py)，图表在 [plotting.py](plotting.py)。

## 2. 下载与运行

安装 Python 3.11 或 3.12 和 Git：

```bash
git clone https://github.com/Xsleet/leo-precise-orbit-demo.git
cd leo-precise-orbit-demo
python -m venv .venv
```

Windows 命令提示符执行 `.venv\Scripts\activate.bat`；Linux/macOS 执行 `source .venv/bin/activate`。
PowerShell 可直接将下文 python 换成 `.\.venv\Scripts\python.exe`。

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

macOS 的 PyTorch 安装命令改为 `python -m pip install torch==2.8.0`，运行时选 cpu。
首次自动下载约 1.32 GB 模型。Numba 首次编译和 8 小时轨道拟合需要一些时间，终端会输出进度。
**GPU 只加速 TimesFM，动力学拟合仍在 CPU，整体运行时间未必显著缩短。**

不下载大模型，只看动力学和线性对照：

```bash
python -m pip install -r requirements.txt
python main.py --model physics
```

## 3. 看懂结果

![过去6小时误差、预热、未来60分钟](data/example.png)

前三行依次是沿轨、法向、径向误差：**过去 6 小时 → 蓝色预热区 → 起报虚线 → 未来 60 分钟**。
黑色点线是留后的参考误差，只用于事后评分；最下面是真正的三维位置误差。

| 方法 | CPU 整段 RMS（m） | GPU 整段 RMS（m） | GPU 60 min RMSE（m） |
|---|---:|---:|---:|
| 动力学（拟合至起报前 20 min） | 5.871 | 5.871 | 6.958 |
| 动力学＋线性误差 | 11.050 | 11.050 | 17.274 |
| 动力学＋TimesFM 误差 | 0.858 | 0.858 | 1.674 |
| 动力学（用最新历史重新拟合） | 4.798 | 4.798 | 6.131 |

TimesFM 的整段 RMS 相对同一动力学轨道降低 **85.4%**，改善 **8/8** 个窗口。
“最新历史重新拟合”利用截至起报时的观测，是比只延续旧轨道更强的动力学对照。

与论文对应的 **60 min 三维 RMSE**：本例 **6.958 → 1.674 m**；
Li 等（2025）为 **0.147 → 0.060 m**。动力学配置、年份、训练方式和测试规模不同，不能据相对降幅给网络排名。
完整逐时长、逐窗口结果见 [RESULTS.md](RESULTS.md)。

## 4. 自己运行全部窗口

```bash
python main.py --device cpu --window 3
python compare_windows.py --device cpu
python compare_windows.py --device gpu
```

八个窗口为 2025 年 6 月 2–3 日每天 00、06、12、18 时（GPS）。也可加 `--gravity j2` 查看低阶动力学对照。
已有模型文件时加 `--checkpoint 模型目录`。首次运行推荐默认窗口 1。

结果在 `outputs/li2025-egm2008-timesfm-cpu/` 或对应 gpu 目录：

| 文件 | 用途 |
|---|---|
| window-01/result.png | 历史误差与未来预报图 |
| window-01/history.csv | 历史输入、残差、预热和输入标记 |
| window-01/prediction.csv | 未来参考与各方法预测坐标 |
| window-01/run.json | 拟合参数、时间边界、模型版本 |
| summary.csv / by_window.csv | 整体与逐窗口成绩 |
| by_lead.csv / paper_comparison.csv | 各预报时长 RMSE 与论文对照 |

真实 CPU/GPU 运行归档在 [reference/li2025](reference/li2025/README.md)。
数据与假设见 [DATA_SOURCES.md](DATA_SOURCES.md)，实验设置与修改记录见 [EXPERIMENT.md](EXPERIMENT.md)。
`python -m unittest discover -v` 运行无需 TimesFM 权重的检查。

## 5. 给学生的三个问题

为什么要留出 20 分钟预热？为什么线性外推会错过周期变化？为什么更大的相对改善不等于更高的绝对精度？

本例是后处理精密产品的离线回放，未模拟发布延迟；简化动力学尚缺潮汐、日月引力、物理阻力、太阳辐射压和精密 EOP。
自编代码 MIT；数据与模型遵循各自许可。TimesFM 3.0 仅用于其许可允许的非商业、非生产教学研究，见 [模型许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)。

同系列：[精密钟差预报](https://github.com/Xsleet/gracefo-clock-prediction) · [空间碎片轨道预测](https://github.com/Xsleet/space-debris-orbit-demo)
