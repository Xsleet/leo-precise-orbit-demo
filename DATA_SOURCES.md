# 数据、动力学与评价约定

## 参考数据

[data/orbit-2019.csv](data/orbit-2019.csv) 保存 2019 年 1 月 1–2 日的 C 星轨道，
以 10 s 采样，共 17,280 条质量标志为零的记录。下载自 GFZ 的 JPL RL04 官方镜像，
下载地址、压缩包及原始 GNV1B 哈希见 [data/orbit-2019.sources.json](data/orbit-2019.sources.json)，
抽取规则和 CSV 哈希见 [data/orbit-2019.manifest.json](data/orbit-2019.manifest.json)。
论文说明使用 GNV1B，但没有明确给出 release 编号；不能确认其原始文件与本例 RL04 完全相同。
仅使用整 10 秒历元，不插值。拟合采用 10 s，TimesFM 和评分采用 60 s。

起报点预先固定为 2019-01-02 00:00 至 07:00 GPS，每小时一次；全部八个窗口保留。
拟合时长为 1/2/4/6/8/12/24 h，TimesFM 输入为 1/2/4/6/8 h，输入长度不超过拟合时长。
所有拟合截至起报点，未来 60 个分钟点在预测完成后用于评分。

位置和速度以 ECEF 表达，单位为 m 和 m/s。时间为自 2000-01-01 12:00:00 GPS 起的连续秒。
参考轨道是事后用 GNSS 解算的 GNV1B 精密轨道产品。数据 DOI 为 [10.5067/GFJPL-L1B04](https://doi.org/10.5067/GFJPL-L1B04)。

力模型见 [DYNAMICS.md](DYNAMICS.md)，辅助数据哈希在 [data/forces/manifest.json](data/forces/manifest.json)。

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
横向对比使用相同的起报窗口及预报时长。
**整段 RMS**：对所有窗口、所有 60 个未来点的误差平方统一求均值，再开方。

统一精度表见 [仓库首页](README.md#预报精度)。运行脚本可产生逐点坐标、历史残差和拟合参数。
[Li 等（2025）](https://doi.org/10.1007/s00190-025-01999-7) 使用 2019 年全年数据、更完整的动力学和专门训练的 GRU Seq2Seq。
本例为 2019 年八个窗口的离线回放，使用原始 TimesFM，未模拟精密产品发布延迟。

## 许可与引用

自编代码适用 MIT；NASA/JPL 数据、NGA 重力场系数和 TimesFM 权重遵循各自规则。
保留数据与模型来源，参见 [PO.DAAC 引用说明](https://podaac.jpl.nasa.gov/CitingPODAAC)。
仓库不分发模型权重或论文 PDF。TimesFM 3.0 用于非商业、非生产教学研究，详见
[官方模型许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)。
