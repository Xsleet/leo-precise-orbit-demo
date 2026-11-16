# 低轨卫星精密轨道预测 Demo

用 **GRACE-FO C 星的真实 GNV1B 精密轨道产品**，演示：

**过去 6 小时 → 起报虚线 → 未来 60 分钟预测**

比较 **原始 TimesFM、二体模型、J2 模型**，支持 CPU/GPU，附带数据和 8 个固定窗口的完整结果。先读 [main.py](main.py) 看流程，再读约 50 行的 [orbit.py](orbit.py) 看动力学。

这里“精密”指参考轨道产品与预测任务，**不表示本例已经实现厘米级轨道预报**。本实验的原始 TimesFM 明显弱于 J2，保留结果供学生分析。

## 1. 下载与运行

安装 Python **3.11 或 3.12** 和 Git：

```bash
git clone https://github.com/Xsleet/leo-precise-orbit-demo.git
cd leo-precise-orbit-demo
python -m venv .venv
```

激活环境：Windows 命令提示符执行 `.venv\Scripts\activate.bat`；Linux/macOS 执行 `source .venv/bin/activate`。PowerShell 可直接将下文 `python` 换成 `.\.venv\Scripts\python.exe`。

**选项 A：CPU，普通电脑即可**

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-timesfm.txt
python main.py --device cpu
```

**选项 B：GPU，需要 NVIDIA 显卡和支持 CUDA 12.8 的驱动**

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-timesfm.txt
python main.py --device gpu
```

两种方式运行相同模型和数据，仅 TimesFM 推理设备不同，二体/J2 均由 CPU 计算。首次自动下载约 **1.32 GB** 模型，无需训练。macOS 的 CPU 安装将第一条命令换为 `python -m pip install torch==2.8.0`。[PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v280)。

想先快速跑通、不下载大模型：

```bash
python -m pip install -r requirements.txt
python main.py --model physics
```

这会只比较二体与 J2。教学部署是本地命令行运行，无需服务器。

## 2. 看懂结果

打开 `outputs/timesfm-cpu/window-01/result.png`；GPU 对应 `timesfm-gpu`。

![过去六小时与未来一小时轨道预测](data/example.png)

上图展示 Z 坐标的周期变化：灰色为历史参考，竖虚线是起报点，浅黄色为未来，黑色点线是留后参考。**下图使用全部 XYZ 计算三维误差**，纵轴为对数；轨迹看起来重合，不代表位置误差很小。

| 模型 | 输入与做法 |
|---|---|
| TimesFM | 过去 360 个 XYZ 位置，三个坐标分别作为时间序列预测；官方预训练模型，未微调 |
| 二体 | 起报时刻的位置、速度，只考虑地球中心引力 |
| J2 | 起报时刻的位置、速度，在中心引力上加入地球扁率的主要影响 |

位置单位为 m、速度为 m/s，采样间隔 60 s，坐标系为 **ECEF 地固系**。物理模型额外使用末点速度，TimesFM 只读位置；这是不同方法的工程对比，并非相同输入特征的模型排名。

默认窗口 1：**2025-06-02 00:00 GPS** 起报。未来 60 个点的三维 RMS，单位 m：

| 模型 | CPU | GPU |
|---|---:|---:|
| 二体 | 32,026.550 | 32,026.550 |
| J2 | **51.679** | **51.679** |
| TimesFM（未微调） | 383,138.457 | 383,137.295 |

`prediction.csv` 保存未来 60 个历元、参考 XYZ、各模型预测 XYZ 和三维误差；`metrics.csv` 保存 RMS 与第 60 分钟误差；`run.json` 记录窗口、时标和模型版本。CPU/GPU 的浮点结果可能略有差异。

```text
每个点的三维误差 = sqrt(dx² + dy² + dz²)
RMS(3D) = sqrt(所有未来点的三维误差平方的平均值)
```

## 3. 换窗口，自主分析

窗口 1–8 依次对应 **6 月 2–3 日每天 00、06、12、18 时（GPS）**。每个窗口固定取过去 360 点（-359 至 0 分钟），预测 +1 至 +60 分钟；不插值、不按结果筛选。

```bash
python main.py --device cpu --window 4
python compare_windows.py --device cpu
python compare_windows.py --device gpu
```

`compare_windows.py` 只加载一次模型，生成所有窗口图、`by_window.csv`、`summary.csv` 和 `comparison.png`。也支持 `--model physics`。

![八个固定窗口的比较](data/comparison.png)

全部 **8 个窗口、480 个未来点**合并后的三维 RMS（m）：

| 模型 | CPU | GPU |
|---|---:|---:|
| 二体 | 36,568.996 | 36,568.996 |
| J2 | **355.724** | **355.724** |
| TimesFM（未微调） | 504,960.234 | 504,960.012 |

本样本中 J2 在所有窗口都最好。直接预测大幅度 XYZ 序列的 TimesFM 没有轨道动力学约束，这提供了一个可讨论的失败案例；不能仅据这 8 个窗口推断其在所有轨道任务中的表现。

给学生的三个问题：为什么同一模型换窗口后误差会变化？为什么只看 Z 轨迹会误判？物理模型少了哪些作用力，还能怎样改进？扩展实验应先确定方案，再保留全部窗口结果。

## 4. 数据与复现

- 数据：2025-06-01 至 03 的 GNV1B RL04 C 星记录，每 60 秒抽取一个质量标志为 0 的点，共 4,320 点。**GPS 时间不是 UTC**。
- 来源、单位、提取命令和简化动力学说明见 [DATA_SOURCES.md](DATA_SOURCES.md)。
- CPU/GPU 的逐点预测及环境记录见 [reference](reference/README.md)；无需下载模型也能核算误差。
- 执行 `python -m unittest discover -v` 检查解析圆轨道、J2 势函数梯度、旋转坐标系、时间边界和归档误差。GitHub Actions 检查 Windows/Linux、Python 3.11/3.12 的基础流程，不在线下载大模型。

这是离线精密产品回放。二体/J2 使用简化动力学，未包括完整高阶重力、大气阻力、日月引力、太阳辐射压及精密地球定向参数。

本仓库自编代码为 MIT；TimesFM、数据遵循各自许可。TimesFM 3.0 用于非商业、非生产的教学研究，权重由用户从官方地址下载，仓库不分发权重。详情见 [模型许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)。

同系列：[精密钟差预报](https://github.com/Xsleet/gracefo-clock-prediction) · [空间碎片轨道预测](https://github.com/Xsleet/space-debris-orbit-demo)
