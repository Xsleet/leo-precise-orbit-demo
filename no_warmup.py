"""无预热：拟合截至起报时刻，用这条拟合轨道的历史残差预报未来误差。"""
import contextlib
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from main import ROOT, forecast_error
from rdop import frames, to_swr, to_xyz
from short_2019 import FIRST, DATA_PATH
from dynamics_cache import fit_cached

FIT_HOURS = (1, 2, 4, 6, 8, 12, 24)
CONTEXT_HOURS = (1, 2, 4, 6, 8)


def grid(data, first, last, step):
    expected = np.arange(first, last + step, step)
    index = np.searchsorted(data[:, 0], expected)
    if (np.any(index >= len(data)) or not np.array_equal(data[index, 0], expected)
            or not np.isfinite(data[index]).all()):
        raise ValueError('观测缺测、时间不连续或数值无效，不插值或替换窗口')
    return data[index]


def history(data, number, fit_hours):
    if number not in range(1, 9) or fit_hours not in FIT_HOURS:
        raise ValueError(f'使用八个固定窗口及以下拟合长度（h）：{FIT_HOURS}')
    origin = FIRST + (number - 1) * 3600
    return grid(data, origin - fit_hours * 3600, origin, 10)


def future_reference(data, number):
    origin = FIRST + (number - 1) * 3600
    return grid(data, origin + 60, origin + 3600, 60)


def prepare(past, recompute=False):
    """只接收历史；返回未来动力学位置和历史残差，不接收未来参考值。"""
    if (past.ndim != 2 or past.shape[1] != 7 or len(past) < 2 or not np.isfinite(past).all()
            or not np.all(np.diff(past[:, 0]) == 10)):
        raise ValueError('需要连续、有限的 10 s 历史位置速度数据')
    span = (past[-1, 0] - past[0, 0]) / 3600
    if span not in FIT_HOURS:
        raise ValueError(f'拟合长度（h）须为 {FIT_HOURS} 之一')
    trajectory, fit = fit_cached(past, 60, recompute=recompute)
    observed = past[::6]
    basis = frames(trajectory)
    residual = to_swr(observed[:, 1:4] - trajectory[:len(observed), :3], basis[:len(observed)])
    return dict(baseline=trajectory[-60:, :3], future_basis=basis[-60:], residual=residual,
                observed=observed, trajectory=trajectory, fit=fit, fit_hours=int(span))


def correct(arc, evaluator, context_hours):
    if context_hours not in CONTEXT_HOURS or context_hours > arc['fit_hours']:
        raise ValueError('输入历史必须位于拟合弧内')
    error = forecast_error(arc['residual'][-context_hours * 60:], evaluator)
    return arc['baseline'] + to_xyz(error, arc['future_basis'])


def prefit(job):
    """独立 CPU 进程预计算一次动力学；不同模型输入长度共享该结果。"""
    number, hours, recompute = job
    past = history(np.loadtxt(DATA_PATH, delimiter=',', skiprows=1), number, hours)
    folder = ROOT/'outputs/window-sweep-prefit'
    folder.mkdir(parents=True, exist_ok=True)
    log = folder/f'fit-{hours:02d}h-window-{number:02d}.log'
    start = time.perf_counter()
    with log.open('w', encoding='utf8') as stream, contextlib.redirect_stdout(stream):
        print(f'实际拟合弧：{hours} h，{len(past)} 个 10 s 观测。底层旧版日志的 8 h 字样为固定提示。')
        _, fit = fit_cached(past, 60, recompute=recompute)
    return dict(window=number, fit_hours=hours, seconds=time.perf_counter()-start,
                fit_rms_3d_m=fit['fit_rms_3d_m'], cache_key=fit['cache_key'],
                reused_cache='cache_origin' in fit)


def fingerprint(names):
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def save_arc(folder, arc):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'history.npz', gps_seconds=arc['observed'][:, 0],
                        reference_xyz=arc['observed'][:, 1:4],
                        fitted_xyz=arc['trajectory'][:-60, :3], residual_swr=arc['residual'])
    (folder/'fit.json').write_text(json.dumps(arc['fit'],indent=2)+'\n',encoding='utf8')
