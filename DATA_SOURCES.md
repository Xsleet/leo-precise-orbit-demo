# 数据、坐标与实验约定

## 数据来自哪里

使用 NASA/JPL **GRACE-FO Level-1B RL04** 的 **GNV1B** C 星产品，原始数据的产品 DOI 为 [10.5067/GFJPL-L1B04](https://doi.org/10.5067/GFJPL-L1B04)。ASCII 集合入口：[PO.DAAC GRACEFO_L1B_ASCII_GRAV_JPL_RL04](https://podaac.jpl.nasa.gov/dataset/GRACEFO_L1B_ASCII_GRAV_JPL_RL04)。

原始文件为 `GNV1B_2025-06-01_C_04.txt`、`GNV1B_2025-06-02_C_04.txt`、`GNV1B_2025-06-03_C_04.txt`。本仓库只附带逐分钟的小样本，不要求学生下载完整日文件。官方格式见 [GRACE-FO Level-1 数据手册](https://isdc-data.gfz.de/grace-fo/DOCUMENTS/Level-1/GRACE-FO_L1_Data_Product_User_Handbook_20190911.pdf)。

这里的位置和速度是 **GPS 定轨得到的后处理轨道产品**，不是原始 GPS 观测量。后处理产品回放不等于实时系统测试；本例未模拟产品发布延迟，也未重新解算定轨。

## 提取方式

[advanced/prepare_data.py](advanced/prepare_data.py) 跳过 YAML 头，读取官方 16 列文本：

| 内容 | 原文件列（从 1 开始） |
|---|---|
| GPS 秒 | 1 |
| 卫星 C / 坐标 E | 2 / 3 |
| XYZ 位置（m） | 4–6 |
| XYZ 速度（m/s） | 10–12 |
| 二进制质量标志 | 16 |

只保留 GPS 秒能被 60 整除、质量标志等于 0 的记录；不平均、不拟合、不插值。当前样本为 4,320 点，没有因标志丢掉抽样点。原文件与抽样 CSV 的 SHA-256 保存在 [data/orbit.manifest.json](data/orbit.manifest.json)。

如已有官方文件，可重建：

```bash
python advanced/prepare_data.py data/raw/GNV1B_2025-06-01_C_04.txt data/raw/GNV1B_2025-06-02_C_04.txt data/raw/GNV1B_2025-06-03_C_04.txt
```

下载完整官方产品可能需要 NASA Earthdata 登录。保留 NASA/JPL 数据来源，引用遵循 [PO.DAAC 引用说明](https://podaac.jpl.nasa.gov/CitingPODAAC)。

## 时间和坐标

- `gps_seconds` 是从 **2000-01-01 12:00:00 GPS** 起的连续秒，与原始文件头一致。不是 UTC，也不是接收机钟时间。
- `x_m,y_m,z_m` 为 **ECEF 地固坐标**，`vx_m_s,vy_m_s,vz_m_s` 为同一地固系中的速度。
- 8 个起报时刻按 GPS 时间定义；代码直接使用连续秒，不做不必要的 UTC 转换。
- 360 个历史点覆盖 `(-6 h, 0]`：从 -359 min 到 0 min。目标为 +1 到 +60 min，共 60 点。

## 简化动力学

[orbit.py](orbit.py) 实现中心引力、可选 J2 和 RK4 积分，内部步长 5 s，输出间隔 60 s。ECEF 是旋转坐标系，必须加入：

```text
a_ECEF = a_gravity - 2 Ω×v - Ω×(Ω×r)
```

地球旋转角速度取 7.292115e-5 rad/s，GM=3.986004418e14 m³/s²，赤道半径=6378136.6 m，J2=1.0826359e-3。常数参考 [IERS 常数表](https://hpiers.obspm.fr/eop-pc/models/constants.html)。

没有计算完整地球重力场、阻力、日月引力、太阳辐射压、极移或精密 EOP。该模型用于教学，精度不能等同于完整轨道预报软件。测试中的厘米级解析解吻合只说明数值积分正确，**不是实际预报精度**。

## TimesFM 与评分

官方模型 [google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch)，未微调。源码固定到 `e51928e27119cb17bebc005be2696b75e0a9e688`；权重固定到 `43046b85ec22d584a13f8098c2ed39c889e129c2`。

两个分支共享同一个原始模型，没有训练或微调：

- 直接预测：XYZ 独立输入，在 float64 中减末值，从 m 转 km，再转 float32；输出还原为 m。
- 误差预测：先从历史末点状态用 J2 回算 -1 至 -359 min，倒序后加上起报点，计算 `GNV1B - J2`。三个误差分量减末值后以 m 输入，预测结果加回未来 J2 位置。

正向与反向使用同一 J2 动力学、同一起报状态和 5 s RK4 步长。因此起报点历史误差为零，没有将“六小时前初始化的误差”错误地加到“起报时重置的新轨道”上。回算会使用历史末点的已知状态，但绝不使用起报后的 GNV1B；这属于起报时可进行的历史残差构造。

物理模型使用末点位置速度；TimesFM 直接分支只使用历史位置；混合分支还使用物理模型和历史误差。因此这是工程方法比较，不是相同输入特征的纯网络排名。官方内部归一化与去趋势保持默认，不按未来误差调参。

预测函数只接受 360 点历史。未来 60 点只在输出完成后评分。保存 `history.csv` 供检查残差来源；`prediction.csv` 保存预测坐标；全部窗口等长，合并 RMS 为各窗口 RMS 平方均值的平方根。

本次 CPU/GPU 均重新运行全部八个窗口，含失败窗口。旧版只有三个方法的归档已被本次四方法结果替换，旧结果仍可通过 Git 历史查看。

MIT 只适用于自编代码；原始数据、依赖和模型有各自许可。TimesFM 权重不随仓库发布。
