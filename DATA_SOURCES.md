# 数据、动力学与评价约定

## 参考数据

使用 NASA/JPL **GRACE-FO Level-1B RL04** 的 **GNV1B** C 星产品。产品 DOI：[10.5067/GFJPL-L1B04](https://doi.org/10.5067/GFJPL-L1B04)；
[PO.DAAC 集合入口](https://podaac.jpl.nasa.gov/dataset/GRACEFO_L1B_ASCII_GRAV_JPL_RL04)。

原始文件为 GNV1B_2025-06-01_C_04.txt、GNV1B_2025-06-02_C_04.txt、GNV1B_2025-06-03_C_04.txt。
仓库附带 4,320 个逐分钟位置、速度样本，不需要学生登录数据网站。
格式依据 [GRACE-FO Level-1 数据手册](https://isdc-data.gfz.de/grace-fo/DOCUMENTS/Level-1/GRACE-FO_L1_Data_Product_User_Handbook_20190911.pdf)。

[advanced/prepare_data.py](advanced/prepare_data.py) 读取 GPS 秒、ECEF 位置和速度，只保留整分钟且质量标志为零的记录；不插值、平均或拟合。
数据及原文件 SHA-256 见 [data/orbit.manifest.json](data/orbit.manifest.json)。

这是真实观测解算得到的后处理精密轨道产品，不是原始 GNSS 观测。本例未重新定轨，也未模拟产品的发布延迟。

## 时间、窗口和输入边界

- 时间是从 2000-01-01 12:00:00 GPS 起的连续秒，不是 UTC。
- 起报时刻固定为 2025-06-02、03 日每天 00、06、12、18 时，共八个窗口，不按结果选择。
- 位置和速度都在 ECEF，单位为 m、m/s。
- 总历史为 -500 到 0 min，共 501 点。
- 动力学拟合只使用 -500 到 -20 min，共 481 点，跨度 8 h。
- 预热为 -19 到 0 min，共 20 点；不参与上述动力学拟合。
- TimesFM 输入为 -359 到 0 min，共 360 点。
- 未来为 +1 到 +60 min，共 60 点，预测完成后才用于评分。
- 最新历史动力学对照重新拟合 -480 到 0 min，使用与其他方法一样已经可用的历史范围。

模型在历史段估计参数；预热和未来使用同一组参数连续积分，起报点不重置状态。未来坐标变换也由动力学预测状态建立。

## 动力学

**当前默认 `--gravity precise`：** Orekit、EGM2008 150 阶/次、DE405、固体潮/极潮、EOT11a 海潮、DTM2000 阻力、宏观太阳辐射压、相对论以及精密 GCRF/ITRF 转换。数据来源、参数、简化项和空间天气可用性逐项列在 [DYNAMICS.md](DYNAMICS.md)，所有辅助文件哈希在 [data/forces/manifest.json](data/forces/manifest.json)。

以下保留旧版 `--gravity egm2008` 的说明，便于复现实验升级前的结果。

旧版采用 **EGM2008 120 阶/次静态重力场 + 地球匀速自转 + 经验加速度**。
EGM2008 原始文献：[Pavlis 等，2012](https://doi.org/10.1029/2011JB008916)。
附带系数来自 [Brahe 固定提交的 EGM2008_120.gfc](https://github.com/duncaneddy/brahe/blob/a5d8988955d477b92453eaf6e565ec64541fcdbf/data/gravity_models/EGM2008_120.gfc)，
来源及 SHA-256 见 [data/gravity.manifest.json](data/gravity.manifest.json)。原始全球模型由 NGA 发布，可通过 [ICGEM](https://icgem.gfz.de/tom_longtime) 获取。

系数为完全归一化、tide-free。计算使用文件对应 GM=3.986004415e14 m³/s²、半径=6378136.3 m。
[gravity.py](gravity.py) 实现归一化球谐递推和势函数梯度；Numba 只负责加速数值计算。

经验加速度有五个系数：沿轨常数项、沿轨一次/圈正余弦项、法向一次/圈正余弦项。
与初始六维状态一起，通过历史位置的最小二乘拟合估计。
沿轨常数项只是吸收部分未建模受力，不能称为物理大气阻力模型或论文中的 Cd 参数。

[rdop.py](rdop.py) 使用 DOP853 积分，最大步长 60 s，rtol=2e-12，atol=1e-7。
ECEF 中加入科里奥利和离心项；自转角速度取 7.292115e-5 rad/s。
沿轨/法向基由位置与惯性速度方向建立，后者为 ECEF 速度加 Ω×r。

这个旧版分支不包含日月引力、潮汐、物理大气密度与阻力、太阳辐射压、极移或精密 EOP。
当前默认已经补充这些项，但与论文仍有 DTM94/DTM2000 等差异。
也可通过 --gravity j2 运行低阶引力对照。J2 常数见 [IERS 常数表](https://hpiers.obspm.fr/eop-pc/models/constants.html)。

## TimesFM

[google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch)，没有微调。
上游源码提交固定到 e51928e27119cb17bebc005be2696b75e0a9e688，
权重 revision 固定到 43046b85ec22d584a13f8098c2ed39c889e129c2。

输入为参考减动力学的 S/W/R 误差。三个分量分别在 float64 中减末值，然后转 float32，以米输入。
预测后加回末值，再由未来动力学基变换回 ECEF，加到同一条动力学轨道上。
不使用未来参考调尺度、去趋势、选窗口或选择模型；官方内部归一化与去趋势保持默认。

模型输入只有误差序列。Li 等的模型还使用位置、速度作为特征，并针对轨道误差专门训练，这两种学习方式并不相同。

## 评分和论文对照

**逐预报时刻的三维 RMSE**：对八个窗口在同一预报分钟的三维误差平方求均值，再开方。
该指标用于与论文 Table 5 的 10/20/30/60 min 对比。
**整段 RMS**：对所有窗口、所有 60 个未来点的误差平方统一求均值，再开方。

同时保存逐窗口成绩、逐点预测坐标、历史误差、拟合参数和源代码哈希，不能只凭曲线是否重合判断精度。
[Li 等（2025）](https://doi.org/10.1007/s00190-025-01999-7) 使用 2019 年全年数据、更完整的动力学和专门训练的 GRU Seq2Seq。
本例使用 2025 年两天的八个预报窗口与原始 TimesFM，是小样本方法演示，绝对精度不是同条件算法排名。

## 许可与引用

自编代码适用 MIT；NASA/JPL 数据、NGA 重力场系数和 TimesFM 权重遵循各自规则。
保留数据与模型来源，参见 [PO.DAAC 引用说明](https://podaac.jpl.nasa.gov/CitingPODAAC)。
仓库不分发模型权重或论文 PDF。TimesFM 3.0 用于非商业、非生产教学研究，详见
[官方模型许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)。
