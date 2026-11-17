"""教学版约化动力学：拟合初始状态及经验加速度，然后连续向未来积分。

默认采用 EGM2008 120 阶/次（可选 J2），加沿轨常数项和沿轨/法向一次/圈正余弦项。
这不是 Li (2025) 的完整力模型；仍缺潮汐、日月引力和物理阻力等。
"""
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from orbit import MU, R_EARTH, J2, OMEGA
from gravity import acceleration as harmonic_gravity


@njit(cache=True)
def frame(state):
    """列向量依次为沿轨 S、法向 W、径向 R，在 ECEF 中表达。"""
    r, v = state[:3], state[3:] + np.cross(OMEGA, state[:3])
    radial = r / np.linalg.norm(r)
    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal)
    along = np.cross(normal, radial)
    return np.column_stack((along, normal, radial))


@njit(cache=True)
def rhs(t, state, empirical, high_order=True):
    r, v = state[:3], state[3:]
    radius = np.linalg.norm(r)
    q = 5 * (r[2] / radius)**2
    gravity = -MU * r / radius**3 + 1.5 * J2 * MU * R_EARTH**2 / radius**5 * np.array(
        [r[0] * (q - 1), r[1] * (q - 1), r[2] * (q - 3)])
    if high_order:
        gravity = harmonic_gravity(r)
    basis = frame(state)
    # u 是从升交点量起的纬度幅角；本例是近极轨，不涉及赤道轨道奇点。
    node = np.array([-basis[1, 1], basis[0, 1], 0.])
    node /= np.linalg.norm(node)
    cos_u = np.dot(basis[:, 2], node)
    sin_u = np.dot(basis[:, 2], np.cross(basis[:, 1], node))
    a_s = empirical[0] + empirical[1] * sin_u + empirical[2] * cos_u
    a_w = empirical[3] * sin_u + empirical[4] * cos_u
    acceleration = gravity + a_s * basis[:, 0] + a_w * basis[:, 1]
    acceleration -= 2 * np.cross(OMEGA, v) + np.cross(OMEGA, np.cross(OMEGA, r))
    return np.concatenate((v, acceleration))


def integrate(initial, empirical, seconds, rtol=2e-12, high_order=True):
    """DOP853 自适应积分；seconds 从 0 开始，包含初始点。"""
    solution = solve_ivp(rhs, (0., float(seconds[-1])), initial, args=(empirical, high_order),
                         method="DOP853", t_eval=seconds, rtol=rtol, atol=1e-7, max_step=60.)
    if not solution.success or not np.isfinite(solution.y).all():
        raise ValueError("动力学积分失败：" + solution.message)
    return solution.y.T


def fit_and_predict(observed, future_points, high_order=True):
    """只接收拟合段的观测；future_points 个后续点完全由积分产生。"""
    times = observed[:, 0] - observed[0, 0]
    initial = observed[0, 1:7]
    # 参数单位分别为 1 m、0.001 m/s、1e-7 m/s²，改善最小二乘数值条件。
    scale = np.r_[np.ones(3), np.full(3, .001), np.full(5, 1e-7)]

    def states(parameters, epochs):
        physical = parameters * scale
        return integrate(initial + physical[:6], physical[6:], epochs, high_order=high_order)

    def residual(parameters):
        return (states(parameters, times)[:, :3] - observed[:, 1:4]).ravel()

    def jacobian(parameters):
        # 固定差分步长，避免对百万米坐标使用过小的数值差分。
        center = residual(parameters)
        return np.column_stack([residual(parameters + unit) - center for unit in np.eye(11)])

    fit = least_squares(residual, np.zeros(11), jac=jacobian, x_scale="jac",
                        max_nfev=60, ftol=1e-6, xtol=1e-4, gtol=1e-4)
    if not fit.success:
        raise ValueError("历史轨道拟合未收敛：" + fit.message)
    all_times = np.arange(len(observed) + future_points) * 60.
    trajectory = states(fit.x, all_times)
    info = {"gravity": "EGM2008 120x120" if high_order else "J2",
            "fit_start_gps_seconds": float(observed[0, 0]),
            "fit_end_gps_seconds": float(observed[-1, 0]), "fit_points": len(observed),
            "fit_rms_3d_m": float(np.sqrt(np.mean(np.sum(fit.fun.reshape(-1, 3)**2, axis=1)))),
            "empirical_m_s2": (fit.x[6:] * scale[6:]).tolist(),
            "initial_state_adjustment": (fit.x[:6] * scale[:6]).tolist(),
            "solver_status": int(fit.status), "function_evaluations": fit.nfev}
    return trajectory, info


def frames(states):
    return np.stack([frame(state) for state in states])


def to_swr(xyz, basis):
    return np.einsum("nij,ni->nj", basis, xyz)


def to_xyz(swr, basis):
    return np.einsum("nij,nj->ni", basis, swr)
