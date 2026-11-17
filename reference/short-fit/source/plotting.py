"""预测完成后才读取未来参考，保存逐点数值、评分和历史到未来的图。"""
import csv
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from main import ROOT, HISTORY, HORIZON, CONTEXT, WARMUP, FIT, REVISION, time_label
from rdop import to_swr

NAMES = {"rdop": "Dynamics (fit ends -20 min)", "rdop_linear": "Dynamics + linear error",
         "rdop_timesfm": "Dynamics + TimesFM error", "fresh_rdop": "Dynamics (refit at 0 min)"}
COLORS = {"rdop": "#64748b", "rdop_linear": "#bc8019",
          "rdop_timesfm": "#bd3764", "fresh_rdop": "#147d78"}


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_result(sample, predictions, details, number, output, device, data_path=None):
    output.mkdir(parents=True, exist_ok=True)
    reference = sample[HISTORY:, 1:4]
    errors = {name: np.linalg.norm(values - reference, axis=1) for name, values in predictions.items()}
    metrics = [{"model": name, "rms_3d_m": float(np.sqrt(np.mean(error**2))),
                "endpoint_3d_m": float(error[-1])} for name, error in errors.items()]
    columns = [sample[HISTORY:, 0], *reference.T]
    header = ["gps_seconds"] + [f"reference_{axis}_m" for axis in "xyz"]
    for name, values in predictions.items():
        columns.extend([*values.T, errors[name]])
        header.extend([f"{name}_{axis}_m" for axis in "xyz"] + [f"{name}_error_3d_m"])
    np.savetxt(output / "prediction.csv", np.column_stack(columns), delimiter=",",
               header=",".join(header), comments="", fmt="%.17g")
    past_base = details["trajectory"][:HISTORY, :3]
    residual = details["past_error_swr"]
    minutes = np.arange(1 - HISTORY, 1)
    context_points = details.get("context_points", CONTEXT)
    columns = np.column_stack([sample[:HISTORY, 0], sample[:HISTORY, 1:4], past_base, residual,
                               minutes > -WARMUP, minutes > -context_points])
    header = ["gps_seconds"] + [f"{kind}_{axis}_m" for kind in ("reference", "baseline") for axis in "xyz"]
    header += [f"residual_{axis}_m" for axis in "swr"] + ["is_warmup", "is_timesfm_input"]
    np.savetxt(output / "history.csv", columns, delimiter=",", header=",".join(header), comments="", fmt="%.17g")
    write_csv(output / "metrics.csv", metrics)
    metadata = {"window": number, "origin": time_label(sample[HISTORY - 1, 0]), "device": device,
        "reference": "GRACE-FO C GNV1B RL04", "frame": "ECEF, residual in SWR",
        "fit_minutes": FIT, "warmup_minutes": WARMUP, "context_points": context_points,
        "horizon_minutes": HORIZON, "timesfm_revision": REVISION, "fine_tuned": False,
        "residual_sign": "reference - dynamics; correction added to same dynamics arc",
        "dynamics": details["fit"].get("model", details["fit"]["gravity"] + " + 5 empirical accelerations"),
        "fit": details["fit"], "fresh_fit": details["fresh_fit"],
        "data_sha256": hashlib.sha256((data_path or ROOT / "data/orbit.csv").read_bytes()).hexdigest()}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    future = np.arange(1, HORIZON + 1)
    future_basis = details["basis"][HISTORY:]
    true_error = to_swr(reference - predictions["rdop"], future_basis)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [1, 1, 1, 1.2]})
    for j, (ax, label) in enumerate(zip(axes[:3], ["Along-track S", "Cross-track W", "Radial R"])):
        ax.plot(minutes[-CONTEXT:], residual[-CONTEXT:, j], color="#44536b", lw=1.2,
                label=f"Past error (last {context_points} min are model input)")
        ax.axvspan(-WARMUP, 0, color="#92c8de", alpha=.25, label="20-min warm-up")
        ax.axvspan(0, HORIZON, color="#f5d0a2", alpha=.25)
        ax.axvline(-WARMUP, color="#6b9bac", linestyle=":", lw=1)
        ax.axvline(0, color="black", linestyle="--", lw=1)
        ax.plot(future, true_error[:, j], "k:", label="Held-out reference error")
        for name, values in predictions.items():
            correction = to_swr(values - predictions["rdop"], future_basis)
            ax.plot(future, correction[:, j], color=COLORS[name], lw=1.3, label=NAMES[name])
        ax.set(xlim=(-CONTEXT + 1, HORIZON), ylabel=label + " (m)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=3, loc="best")
    axes[2].set_xlabel("Minutes relative to forecast start | past 6 h -> future 60 min")
    for name, error in errors.items():
        axes[3].plot(future, error, color=COLORS[name], label=NAMES[name])
    axes[3].set(xlim=(1, HORIZON), xlabel="Forecast lead (min)", ylabel="3-D position error (m)")
    axes[3].grid(alpha=.2)
    axes[3].legend(fontsize=8, ncol=2)
    fig.suptitle(f"GRACE-FO C | {metadata['origin']} | window {number} | {device.upper()}\n"
                 "Dynamics + historical-error forecast (TimesFM is not fine-tuned)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "result.png", dpi=150)
    plt.close(fig)
    for row in metrics:
        print(f"{row['model']}: arc RMS={row['rms_3d_m']:.3f} m; 60-min error={row['endpoint_3d_m']:.3f} m", flush=True)
    return errors, metrics
