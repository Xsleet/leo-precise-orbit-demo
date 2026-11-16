"""低轨卫星精密轨道预报：8 h 拟合 -> 20 min 预热 -> TimesFM 修正未来 60 min。"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import numpy as np
from rdop import fit_and_predict, frames, to_swr, to_xyz

ROOT = Path(__file__).resolve().parent
FIT, WARMUP, CONTEXT, HORIZON, STEP, WINDOWS = 480, 20, 360, 60, 60, 8
HISTORY = FIT + WARMUP + 1  # 从 -500 到 0 分钟，共 501 个历史点
MODEL_ID = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def origin_seconds(number):
    return (datetime(2025, 6, 2) - datetime(2000, 1, 1, 12)).total_seconds() + (number - 1) * 6 * 3600


def time_label(seconds):
    return (datetime(2000, 1, 1, 12) + timedelta(seconds=float(seconds))).strftime("%Y-%m-%d %H:%M GPS")


def select_window(data, number):
    if number not in range(1, WINDOWS + 1) or data.ndim != 2 or data.shape[1] != 7:
        raise ValueError("窗口编号或七列位置速度数据格式不正确")
    if not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError("时间必须严格递增")
    expected = origin_seconds(number) + np.arange(1 - HISTORY, HORIZON + 1) * STEP
    start = np.searchsorted(data[:, 0], expected[0])
    sample = data[start:start + len(expected)]
    if not np.array_equal(sample[:, 0], expected) or not np.isfinite(sample).all():
        raise ValueError("窗口存在缺测或无效值，不插值")
    return sample


def forecast_error(errors, evaluator):
    """S/W/R 三轴独立预测，float64 中减末值后再以米输入原始 TimesFM。"""
    inputs = (errors - errors[-1]).T.astype(np.float32)
    outputs = list(evaluator.predict_batch(list(inputs), horizon=HORIZON,
        return_quantiles=False, make_positive=False, use_symmetric_averaging=False))
    result = np.stack([out.forecast for out in outputs], axis=1).astype(np.float64) + errors[-1]
    if result.shape != (HORIZON, 3) or not np.isfinite(result).all():
        raise ValueError("TimesFM 未返回完整、有限的三轴误差预报")
    return result


def predict(history, evaluator=None, gravity="precise", refit=False):
    """只接收历史。未来参考值不进入拟合、坐标系、线性外推或 TimesFM。"""
    if history.shape != (HISTORY, 7) or not np.isfinite(history).all():
        raise ValueError(f"仅接受 {HISTORY} 个历史位置速度点，不接收未来参考")
    if not np.all(np.diff(history[:, 0]) == STEP):
        raise ValueError("历史采样间隔必须为 60 秒")
    # 1. 只拟合至 -20 min；之后连续积分，起报点 0 min 不重置状态。
    if gravity == "precise":
        from dynamics_cache import fit_cached
        fit_arc = lambda past, count: fit_cached(past, count, recompute=refit)
    else:
        fit_arc = lambda past, count: fit_and_predict(past, count, gravity == "egm2008")
    trajectory, fit = fit_arc(history[:FIT + 1], WARMUP + HORIZON)
    basis = frames(trajectory)
    past_error = to_swr(history[:, 1:4] - trajectory[:HISTORY, :3], basis[:HISTORY])
    result = {"rdop": trajectory[HISTORY:, :3]}
    # 2. 简单对照：只对预热段的 20 个实际预报误差拟合直线。
    line = np.polyfit(np.arange(1 - WARMUP, 1), past_error[-WARMUP:], 1)
    linear_error = np.arange(1, HORIZON + 1)[:, None] * line[0] + line[1]
    result["rdop_linear"] = result["rdop"] + to_xyz(linear_error, basis[HISTORY:])
    # 3. TimesFM 看到拟合残差 + 预热预报误差，修正同一条动力学轨道。
    if evaluator is not None:
        correction = forecast_error(past_error[-CONTEXT:], evaluator)
        result["rdop_timesfm"] = result["rdop"] + to_xyz(correction, basis[HISTORY:])
    # 更强的动力学对照：利用截至 0 min 的最新 8 h 数据重新拟合。
    fresh, fresh_fit = fit_arc(history[-(FIT + 1):], HORIZON)
    result["fresh_rdop"] = fresh[-HORIZON:, :3]
    details = {"trajectory": trajectory, "basis": basis, "past_error_swr": past_error,
               "fit": fit, "fresh_fit": fresh_fit}
    return result, details


def load_evaluator(device, checkpoint):
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    if device == "gpu" and not torch.cuda.is_available():
        raise ValueError("未发现 NVIDIA GPU，请使用 --device cpu")
    print("加载原始 TimesFM 3.0，无微调；首次下载约 1.32 GB……", flush=True)
    return TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint, revision=REVISION,
        device="cuda" if device == "gpu" else "cpu", per_core_batch_size=1))


def arguments(batch=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", choices=["timesfm", "physics"], default="timesfm")
    parser.add_argument("--checkpoint", default=MODEL_ID)
    parser.add_argument("--gravity", choices=["precise", "egm2008", "j2"], default="precise")
    parser.add_argument("--refit", action="store_true", help="不复用动力学缓存，重新拟合历史轨道")
    if not batch:
        parser.add_argument("--window", type=int, choices=range(1, WINDOWS + 1), default=1)
    args = parser.parse_args()
    if args.model == "physics" and args.device == "gpu":
        parser.error("动力学模型使用 CPU；GPU 选项用于 TimesFM")
    return args


def main():
    from plotting import save_result
    args = arguments()
    data = np.loadtxt(ROOT / "data/orbit.csv", delimiter=",", skiprows=1)
    sample = select_window(data, args.window)
    evaluator = load_evaluator(args.device, args.checkpoint) if args.model == "timesfm" else None
    predictions, details = predict(sample[:HISTORY], evaluator, args.gravity, args.refit)
    output = ROOT / "outputs" / f"li2025-{args.gravity}-{args.model}-{args.device}" / f"window-{args.window:02d}"
    save_result(sample, predictions, details, args.window, output, args.device)
    print(f"结果：{output}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
