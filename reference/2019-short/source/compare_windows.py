"""八个固定窗口全部评估；按预报时刻计算跨窗口 RMSE，与论文 Table 5 对齐。"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
from main import ROOT, WINDOWS, HISTORY, arguments, select_window, predict, load_evaluator
from plotting import plt, NAMES, COLORS, write_csv, save_result


def summarize(all_errors):
    """不把误差范数的算术平均冒充 RMSE；所有窗口权重相同。"""
    rows, lead_rows = [], []
    for name, values in all_errors.items():
        errors = np.stack(values)  # [window, lead], 每个元素为三维距离
        rms = np.sqrt(np.mean(errors**2, axis=1))
        lead_rmse = np.sqrt(np.mean(errors**2, axis=0))
        rows.append({"model": name, "windows": len(values),
            "pooled_rms_3d_m": float(np.sqrt(np.mean(errors**2))),
            "endpoint_rmse_3d_m": float(lead_rmse[-1]),
            "median_window_rms_3d_m": float(np.median(rms)),
            "worst_window_rms_3d_m": float(rms.max())})
        for minute, value in enumerate(lead_rmse, 1):
            lead_rows.append({"model": name, "lead_minutes": minute, "rmse_3d_m": float(value)})
    return rows, lead_rows


def main():
    args = arguments(batch=True)
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ["main.py", "precise.py", "dynamics_cache.py", "prepare_forces.py", "rdop.py", "gravity.py", "orbit.py", "plotting.py", "compare_windows.py"]}
    start = time.perf_counter()
    data = np.loadtxt(ROOT / "data/orbit.csv", delimiter=",", skiprows=1)
    samples = [select_window(data, number) for number in range(1, WINDOWS + 1)]
    evaluator = load_evaluator(args.device, args.checkpoint) if args.model == "timesfm" else None
    folder = ROOT / "outputs" / f"li2025-{args.gravity}-{args.model}-{args.device}"
    all_errors, window_rows = {}, []
    for number, sample in enumerate(samples, 1):
        print(f"Window {number}/{WINDOWS}", flush=True)
        predictions, details = predict(sample[:HISTORY], evaluator, args.gravity, args.refit)
        errors, metrics = save_result(sample, predictions, details, number,
                                       folder / f"window-{number:02d}", args.device)
        window_rows.extend({"window": number, **row} for row in metrics)
        for name, error in errors.items():
            all_errors.setdefault(name, []).append(error)
    summaries, lead_rows = summarize(all_errors)
    write_csv(folder / "by_window.csv", window_rows)
    write_csv(folder / "summary.csv", summaries)
    write_csv(folder / "by_lead.csv", lead_rows)
    paper_rows = []
    for minute, baseline_cm, hybrid_cm in zip([10, 20, 30, 60], [3., 4.9, 7.3, 14.7], [1., 1.9, 2.9, 6.]):
        row = {"lead_minutes": minute, "paper_rdop_m": baseline_cm / 100,
               "paper_dnn_rdop_m": hybrid_cm / 100}
        for name, errors in all_errors.items():
            row[name + "_m"] = float(np.sqrt(np.mean(np.stack(errors)[:, minute - 1]**2)))
        paper_rows.append(row)
    write_csv(folder / "paper_comparison.csv", paper_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, values in all_errors.items():
        e = np.stack(values)
        axes[0].plot(range(1, WINDOWS + 1), np.sqrt(np.mean(e**2, axis=1)), "o-",
                     color=COLORS[name], label=NAMES[name])
        axes[1].plot(range(1, 61), np.sqrt(np.mean(e**2, axis=0)), color=COLORS[name], label=NAMES[name])
    axes[0].set(xlabel="Fixed window", ylabel="Whole-arc 3-D RMS (m)", xticks=range(1, WINDOWS + 1))
    axes[1].set(xlabel="Forecast lead (min)", ylabel="Across-window 3-D RMSE (m)")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"All {WINDOWS} fixed windows | {args.device.upper()} | {args.gravity} dynamics")
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=150)
    plt.close(fig)
    packages = {}
    for name in ["numpy", "scipy", "numba", "matplotlib", "torch", "timesfm", "orekit-jpype", "JPype1", "jdk4py"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
        "device": args.device, "checkpoint": ("local:" + Path(args.checkpoint).name if Path(args.checkpoint).is_dir() else args.checkpoint), "elapsed_seconds": time.perf_counter() - start,
        "gravity": args.gravity, "source_sha256": source_hashes}
    if args.gravity == "precise":
        metadata["force_manifest_sha256"] = hashlib.sha256((ROOT / "data/forces/manifest.json").read_bytes()).hexdigest()
        metadata["dynamics_execution"] = "CPU; deterministic cached fits shared by CPU/GPU TimesFM runs"
    if evaluator is not None:
        import torch
        metadata["torch_threads"] = torch.get_num_threads()
        metadata["gpu"] = torch.cuda.get_device_name(0) if args.device == "gpu" else None
    (folder / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"全部结果：{folder}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
