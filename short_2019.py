"""2019 短期验证：固定 1 月 2 日 00–07 时，8 个窗口，每次预报 60 min。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import numpy as np
from main import ROOT, HISTORY, WINDOWS, MODEL_ID, load_evaluator, predict
from plotting import save_result, write_csv, plt, NAMES, COLORS
from compare_windows import summarize

DATA_PATH = ROOT / 'data/orbit-2019.csv'
FIRST = (datetime(2019, 1, 2) - datetime(2000, 1, 1, 12)).total_seconds()


def window(data, number):
    """整段数据仅在评分端保留；预测函数只收到起报点及之前的历史。"""
    if number not in range(1, WINDOWS + 1):
        raise ValueError('窗口必须为 1–8')
    origin = FIRST + (number - 1) * 3600
    times = origin + np.arange(-30000, 3601, 10)
    start = np.searchsorted(data[:, 0], times[0])
    dense = data[start:start + len(times)]
    if not np.array_equal(dense[:, 0], times) or not np.isfinite(dense).all():
        raise ValueError(f'窗口 {number} 缺测，不插值、不替换窗口')
    return dense[::6], dense[:3001]


def prefit(job):
    from dynamics_cache import fit_cached
    number, fresh, recompute = job
    _, past = window(np.loadtxt(DATA_PATH, delimiter=',', skiprows=1), number)
    fitted = past[120:] if fresh else past[:2881]
    _, info = fit_cached(fitted, 60 if fresh else 80, recompute=recompute)
    return number, fresh, info['fit_rms_3d_m']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--model', choices=['physics', 'timesfm'], default='timesfm')
    parser.add_argument('--checkpoint', default=MODEL_ID)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=1,
                        help='首次计算动力学时的 CPU 进程数；缓存存在时无需设置')
    parser.add_argument('--refit', action='store_true', help='重新拟合，不复用动力学缓存')
    parser.add_argument('--context', type=int, choices=[80, 360], default=80,
                        help='80 对齐论文的历史长度；360 保留旧 Demo 的 6 小时输入')
    args = parser.parse_args()
    if args.model == 'physics' and args.device != 'cpu':
        parser.error('动力学运行在 CPU，GPU 用于 TimesFM')
    started = time.perf_counter()
    if args.workers > 1:
        jobs = [(n, fresh, args.refit) for n in range(1, WINDOWS+1) for fresh in [False, True]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(prefit, jobs):
                print('历史拟合完成：', result, flush=True)
    folder = ROOT / 'outputs' / f'2019-short-{args.model}-{args.device}-ctx{args.context}'
    data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
    evaluator = load_evaluator(args.device, args.checkpoint) if args.model == 'timesfm' else None
    all_errors, by_window = {}, []
    for number in range(1, WINDOWS+1):
        print(f'Window {number}/{WINDOWS}', flush=True)
        sample, history10 = window(data, number)
        predictions, details = predict(sample[:HISTORY], evaluator, fit_history=history10,
                                       context_points=args.context, refit=args.refit and args.workers==1)
        errors, rows = save_result(sample, predictions, details, number,
                                  folder/f'window-{number:02d}', args.device, DATA_PATH)
        by_window.extend({'window': number, **row} for row in rows)
        for name, values in errors.items():
            all_errors.setdefault(name, []).append(values)
    summary, leads = summarize(all_errors)
    write_csv(folder/'summary.csv', summary)
    write_csv(folder/'by_window.csv', by_window)
    write_csv(folder/'by_lead.csv', leads)
    comparison = []
    for minute, base, hybrid in zip([10,20,30,60], [3,4.9,7.3,14.7], [1,1.9,2.9,6]):
        row = dict(lead_minutes=minute, paper_test_rdop_m=base/100, paper_test_dnn_rdop_m=hybrid/100)
        row.update({name+'_m': float(np.sqrt(np.mean(np.stack(values)[:,minute-1]**2)))
                    for name,values in all_errors.items()})
        comparison.append(row)
    write_csv(folder/'paper_comparison.csv', comparison)
    fig, ax = plt.subplots(figsize=(8,4))
    for name, values in all_errors.items():
        ax.plot(range(1,61), np.sqrt(np.mean(np.stack(values)**2,axis=0)), color=COLORS[name], label=NAMES[name])
    ax.set(xlabel='Forecast lead (min)', ylabel='Across-window 3-D RMSE (m)',
           title='GRACE-FO C | 2019-01-02 00:00–07:00 GPS | 8 hourly windows')
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(folder/'comparison.png', dpi=160); plt.close(fig)
    sources = ['short_2019.py','main.py','precise.py','dynamics_cache.py','plotting.py','compare_windows.py','rdop.py','gravity.py','orbit.py']
    metadata = dict(python=platform.python_version(), platform=platform.platform(), device=args.device,
        context_points=args.context, model=args.model, elapsed_seconds=time.perf_counter()-started,
        data_sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','orekit-jpype','JPype1','jdk4py']},
        windows=8, selection='Fixed before evaluation: 2019-01-02 00 through 07 GPS, hourly; no replacements',
        fit_step_seconds=10, output_step_seconds=60, force_manifest_sha256=hashlib.sha256((ROOT/'data/forces/manifest.json').read_bytes()).hexdigest())
    if evaluator is not None:
        import torch
        metadata.update(torch=torch.__version__, torch_threads=torch.get_num_threads(),
                        gpu=torch.cuda.get_device_name(0) if args.device=='gpu' else None,
                        timesfm=importlib.metadata.version('timesfm'),
                        checkpoint=('local:'+Path(args.checkpoint).name
                                    if Path(args.checkpoint).is_dir() else args.checkpoint))
    (folder/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
    print('结果：',folder,flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8')
    sys.stderr.reconfigure(encoding='utf8')
    main()
