"""CPU/GPU 共用确定性的 CPU 动力学，避免重复拟合；源码/数据/版本变化会自动失效。"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from precise import fit_and_predict, DATA


def fit_cached(observed, future_points, recompute=False):
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256(observed.astype('<f8').tobytes())
    digest.update(str(future_points).encode())
    digest.update((root / 'precise.py').read_bytes())
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name in sorted(manifest):
        raw = (DATA / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[name]['sha256']:
            raise ValueError('力模型数据校验失败：' + name)
        digest.update(raw)
    for name in ['numpy', 'scipy', 'orekit-jpype', 'JPype1', 'jdk4py']:
        digest.update((name + importlib.metadata.version(name)).encode())
    path = root / 'outputs/dynamics-cache' / (digest.hexdigest() + '.npz')
    supplied = root / 'reference/precise/dynamics-cache' / path.name
    short = root / 'reference/2019-short/dynamics-cache' / path.name
    sweep = root / 'reference/window-sweep/dynamics-cache' / path.name
    output_points = int((observed[-1, 0] - observed[0, 0]) / 60) + 1 + future_points
    for candidate in [path, supplied, short, sweep]:
        if candidate.exists() and not recompute:
            with np.load(candidate, allow_pickle=False) as cached:
                info = json.loads(str(cached['info']))
                trajectory = cached['trajectory']
                if info['cache_key'] != digest.hexdigest() or trajectory.shape != (output_points, 6):
                    raise ValueError('动力学缓存与当前实验不匹配')
                if not np.isfinite(trajectory).all():
                    raise ValueError('动力学缓存包含无效值')
                print('  复用相同历史、力模型和软件版本的 CPU 动力学缓存', flush=True)
                info['cache_origin'] = str(candidate.relative_to(root))
                return trajectory, info
    trajectory, info = fit_and_predict(observed, future_points)
    info['cache_key'] = digest.hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, trajectory=trajectory, info=json.dumps(info))
    return trajectory, info
