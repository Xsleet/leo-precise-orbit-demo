"""相同八个窗口、相同权重，分别测 CPU/GPU 的加载和稳定推理时间。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
import numpy as np
from main import ROOT, MODEL_ID, REVISION, load_evaluator, forecast_error
from no_warmup import CONTEXT_HOURS, history, prepare, fingerprint
from short_2019 import DATA_PATH
from plotting import write_csv


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--checkpoint', default=MODEL_ID)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 3 or args.threads < 1:
        parser.error('至少重复三次，线程数必须为正')
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == 'gpu' and not torch.cuda.is_available():
        parser.error('没有可用的 NVIDIA GPU')
    folder = ROOT/'outputs'/f'benchmark-{args.device}'
    folder.mkdir(parents=True, exist_ok=True)

    # 数据准备及动力学不计入 TimesFM 推理时间。
    data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
    residuals = [prepare(history(data, n, 8))['residual'] for n in range(1, 9)]
    input_hash = hashlib.sha256(np.asarray(residuals, dtype='<f8').tobytes()).hexdigest()

    def sync():
        if args.device == 'gpu':
            torch.cuda.synchronize()  # CUDA 异步执行，必须等计算完成后才停表。

    start = time.perf_counter()
    evaluator = load_evaluator(args.device, args.checkpoint)
    sync()
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    forecast_error(residuals[0][-480:], evaluator)
    sync()
    first_seconds = time.perf_counter() - start
    rows, forecasts = [], {}
    for hours in CONTEXT_HOURS:
        contexts = [r[-hours*60:] for r in residuals]
        # 每个输入长度先运行两次；这是计算预热，与轨道的 20 min 预热无关。
        for _ in range(2):
            forecast_error(contexts[0], evaluator)
        sync()
        elapsed = []
        for _ in range(args.repeats):
            sync()
            start = time.perf_counter()
            result = np.array([forecast_error(r, evaluator) for r in contexts])
            sync()
            elapsed.append(time.perf_counter() - start)
        forecasts[f'context_{hours:02d}h'] = result
        rows.append(dict(context_hours=hours, windows=8, median_seconds=statistics.median(elapsed),
                         seconds_per_window=statistics.median(elapsed)/8,
                         **{f'repeat_{i+1}_seconds': t for i, t in enumerate(elapsed)}))
        print(f'{args.device}, 输入 {hours} h：8 个窗口 {statistics.median(elapsed):.3f} s', flush=True)
    write_csv(folder/'timing.csv', rows)
    np.savez_compressed(folder/'forecasts.npz', **forecasts)
    metadata = dict(device=args.device, model=MODEL_ID, revision=REVISION, fine_tuned=False,
        model_load_seconds=load_seconds, first_8h_inference_seconds=first_seconds,
        repeats=args.repeats, warmup_calls_per_length=2, threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), per_core_batch_size=1,
        torch=torch.__version__, timesfm=importlib.metadata.version('timesfm'),
        python=platform.python_version(), platform=platform.platform(),
        cpu=platform.processor(), logical_cpus=os.cpu_count(), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if args.device=='gpu' else None,
        input_sha256=input_hash,
        source_sha256=fingerprint(['benchmark_timesfm.py', 'main.py', 'no_warmup.py', 'dynamics_cache.py']),
        includes='forecast_error: centering, float32 conversion, model inference, CPU/GPU transfers, restored float64 outputs',
        excludes='dynamics, file IO, model loading, warmup',
        inference_warmup_is_not_orbit_warmup=True)
    (folder/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf8')
    print(f'加载 {load_seconds:.3f} s；首次 8 h 输入推理 {first_seconds:.3f} s', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8'); sys.stderr.reconfigure(encoding='utf8')
    run()
