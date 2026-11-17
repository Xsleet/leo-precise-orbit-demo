# 动力学升级：哪些对齐了 Li 等（2025）

学生先看 `no_warmup.py`：**拟合过去 → 算出历史残差 → TimesFM 外推误差 → 加回动力学轨道**。
复杂的力模型交给 Orekit，集中在 `precise.py`；不需要学生自行实现球谐函数或岁差章动。

## 配置对照

| 项目 | Li 等 Table 1 / 最优 V-8 | 当前实现 |
|---|---|---|
| 静态重力场 | EGM2008 150 阶 | EGM2008 150 阶/次 |
| 日月引力 | JPL DE405 | 原始 DE405 二进制星历，日/月第三体引力 |
| 固体潮 | IERS 2010 | Orekit IERS 2010，含极潮 |
| 海潮 | EOT11a | EOT11a 120 阶/次、18 个主分潮，含负荷效应；未加 238 个次分潮的导纳插值 |
| 太阳辐射压 | 宏观模型 | GRACE-FO 手册 8 面模型，地球遮挡；名义 VVLH 姿态 |
| 大气阻力 | DTM94 | **DTM2000 替代**；未取得可直接复现的 DTM94 实现 |
| 阻力参数 | 分段尺度因子 | 从拟合末端向前按名义周期分段，预报沿用最后一个 |
| 经验加速度 | 沿轨 S、法向 W 的 1PR 正余弦 | 同方向、同阶数；采用名义周期的时间谐波 |
| 坐标/时间 | 精密轨道计算 | GCRF↔ITRF，IERS 2010 岁差章动、极移和 UT1；GPS→UTC/TAI 转换 |
| 相对论 | 表中未单列 | Schwarzschild 修正 |
| 拟合 / 预热 / 预报 | 8 h / 20 min / 60 min | 1/2/4/6/8/12/24 h / 0 min / 60 min |
| 动力学观测采样 | 10 s | 10 s |
| 网络 | 专门训练的 GRU Seq2Seq | 原始 TimesFM 3.0，无微调 |
| 网络历史长度 | 1 h 拟合残差 + 20 min 预热误差 | 1/2/4/6/8 h 拟合残差，输入不超过拟合弧 |

**这是按论文配置升级的可复现教学实验，不是完整复现论文。**
2019 短期实验固定使用 1 月 2 日 00–07 时的八个连续起报窗口，每小时一次。
论文使用 2019 全年约 8,600 个窗口，前 80% 训练、后 20% 测试；本例一月短期窗口并非论文的测试集。
空间天气现在按实际 UTC 日期读取，预报仍冻结到拟合截止时刻。
原文把部分“revolution”写作“resolution”；这里把阻力分段具体定义为名义轨道周期，并公布每段参数。
当拟合弧至少一圈，阻力分段与拟合末端对齐，最前面不足一圈的片段并入首段。1 h 拟合短于约 95 min 的轨道周期，采用单个阻力尺度覆盖整段，其估计以及经验加速度外推更依赖这段有限观测。

## 三个容易影响结论的细节

1. **只用过去的数据估计参数。** 初始位置/速度、4 个经验系数、阻力尺度均由历史位置最小二乘估计。所有拟合截止起报点，动力学和 TimesFM 修正均从同一点预测未来 60 min。
2. **不用未来实测空间天气。** F10.7 取前一完整 UTC 日；均值取后向 81 日；Kp 取上一完整 3 h 段和过去 24 h。预报时冻结到该动力学拟合截止时刻。历史资料来自事后整理的 CSSI 档案，未模拟实际发布延迟。
3. **精密 EOP 是离线已知辅助量。** 使用发布后的 EOP 覆盖拟合和预报区间，未模拟实时 EOP 预测误差；它不是未来卫星轨道观测。不能把离线回放成绩直接称为实时精度。

宏观模型采用手册的名义质量 601.214 kg、面元面积和光学系数，省略小型杆件。姿态假设为 +X 沿轨、+Z 对地。阻力尺度可吸收一部分面积和质量误差。

一次/圈经验项使用起始密切轨道的名义周期；真实纬度幅角与匀速相位并不完全相同。以上差异都可能限制分米级目标，不能事先把误差归因于 TimesFM。

## 数值精度与复现

采用 DOP853 积分，最大步长 60 s，相对容差 1e-15，位置/速度绝对容差分别为 1e-7 m、1e-10 m/s。
这些是局部误差控制参数，不代表整段轨道的实际误差就这么小。天气跳变、阻力分段和日食边界均显式交给事件检测处理。

附带的动力学缓存按历史数据、力模型数据、动力学源码及依赖版本校验。使用 `--refit` 可重新拟合并积分。

## 数据来源与核验

- [Li、Wu、Ge（2025），Journal of Geodesy](https://doi.org/10.1007/s00190-025-01999-7)：Table 1、Table 3、Table 5。
- [ICGEM EGM2008](https://icgem.gfz.de/tom_longtime)：截断到 150 阶，原始下载 URL 与 SHA-256 在 `data/forces/manifest.json`。
- [JPL DE405](https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de405/)：`lnx1900.405`。
- [EOT11a 原始球谐系数](https://www.sirgas.org/archive/EOT11a/data/SphericalHarmonics/)及[技术说明](https://www.sirgas.org/archive/EOT11a/doc/TN_EOT11a.pdf)：作者 Rieser、Mayer-Gürr、Savcenko、Bosch、Wünsch、Dahle、Flechtner；数据模型依据其许可与署名要求使用。
- [Orekit Data](https://gitlab.orekit.org/orekit/orekit-data)：固定提交 `2d6edc9118983d95c51303df05db16ada90b7ee2`，包含 IERS EOP、闰秒、CSSI 空间天气。
- [GRACE-FO Level-1 Handbook](https://isdc-data.gfz.de/grace-fo/DOCUMENTS/Level-1/GRACE-FO_L1_Data_Product_User_Handbook_20190911.pdf)：Table 4–5 宏观模型。
- [Orekit 开源动力学库](https://github.com/CS-SI/Orekit)：Apache 2.0；通过 `orekit-jpype` 调用，`jdk4py` 提供 Java 运行时。

EOT11a 的 `.cos.gfc` / `.sin.gfc` 已含负荷与相位改正，直接转换为 Orekit 的无量纲势系数，不再乘负荷因子或增加相位。该目录中的 IERS 文本是 **cm 单位的潮高系数**，不能当作引力势系数直接加载。转换和原始文件哈希均公开。

`python prepare_forces.py` 校验附带力模型；`python prepare_forces.py --download` 从固定来源重建并核验。
