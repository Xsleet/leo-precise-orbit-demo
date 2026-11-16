"""EGM2008 的球谐引力。高阶计算放在这里，学生可先读 main.py。

系数采用完全归一化、无 Condon-Shortley 相位的 Legendre 函数。
公式：V = GM/r * sum((R/r)^n * P_nm(sin(phi)) * (C*cos(m*lon)+S*sin(m*lon)))。
加速度为 V 的梯度；单位 m、s。模型和来源随 data/ 附带。
"""
from pathlib import Path
import numpy as np
from numba import njit

GM, RADIUS, DEGREE = 3.986004415e14, 6378136.3, 120
C = np.zeros((DEGREE + 1, DEGREE + 1))
S = np.zeros_like(C)
for line in (Path(__file__).parent / "data/EGM2008_120.gfc").read_text().splitlines():
    words = line.split()
    if words and words[0] == "gfc":
        n, m = int(words[1]), int(words[2])
        C[n, m], S[n, m] = (float(word.replace("d", "e")) for word in words[3:5])
# 递推系数只计算一次，积分中反复复用。
A, B = np.zeros_like(C), np.zeros_like(C)
for n in range(2, DEGREE + 1):
    for m in range(n - 1):
        A[n, m] = np.sqrt((2*n - 1)*(2*n + 1) / ((n - m)*(n + m)))
        B[n, m] = np.sqrt((2*n + 1)*(n + m - 1)*(n - m - 1) / ((n - m)*(n + m)*(2*n - 3)))


@njit(cache=True)
def acceleration(position, cosine=C, sine=S):
    radius = np.linalg.norm(position)
    sin_phi = position[2] / radius
    cos_phi = np.hypot(position[0], position[1]) / radius
    lon = np.arctan2(position[1], position[0])
    p, dp = np.zeros_like(cosine), np.zeros_like(cosine)
    p[0, 0] = 1.
    if len(cosine) > 1:
        p[1, 0], dp[1, 0] = np.sqrt(3.)*sin_phi, np.sqrt(3.)*cos_phi
        p[1, 1], dp[1, 1] = np.sqrt(3.)*cos_phi, -np.sqrt(3.)*sin_phi
    for n in range(2, len(cosine)):
        k = np.sqrt((2*n + 1)/(2*n))
        p[n, n] = k*cos_phi*p[n-1, n-1]
        dp[n, n] = k*(-sin_phi*p[n-1, n-1] + cos_phi*dp[n-1, n-1])
        k = np.sqrt(2*n + 1.)
        p[n, n-1] = k*sin_phi*p[n-1, n-1]
        dp[n, n-1] = k*(cos_phi*p[n-1, n-1] + sin_phi*dp[n-1, n-1])
        for m in range(n - 1):
            p[n, m] = A[n, m]*sin_phi*p[n-1, m] - B[n, m]*p[n-2, m]
            dp[n, m] = A[n, m]*(cos_phi*p[n-1, m] + sin_phi*dp[n-1, m]) - B[n, m]*dp[n-2, m]
    orders = np.arange(len(cosine))
    cm, sm = np.cos(orders*lon), np.sin(orders*lon)
    radial, north, east = 0., 0., 0.
    for n in range(len(cosine)):
        power = (RADIUS/radius)**n
        for m in range(n + 1):
            term = cosine[n, m]*cm[m] + sine[n, m]*sm[m]
            radial -= (n+1)*power*p[n, m]*term
            north += power*dp[n, m]*term
            east += power*p[n, m]*m*(-cosine[n, m]*sm[m] + sine[n, m]*cm[m])
    # 本例卫星不穿越精确极点；纬度导数向量在 ECEF 中展开。
    er = position/radius
    en = np.array([-sin_phi*np.cos(lon), -sin_phi*np.sin(lon), cos_phi])
    ee = np.array([-np.sin(lon), np.cos(lon), 0.])
    return GM/radius**2 * (radial*er + north*en + east/max(cos_phi, 1e-15)*ee)
