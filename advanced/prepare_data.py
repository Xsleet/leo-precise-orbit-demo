"""从官方 GNV1B C 星日文件提取每分钟的真实位置/速度，保留来源哈希。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--step", type=int, choices=[10, 60], default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "data/orbit.csv")
    args = parser.parse_args()
    records, sources = {}, []
    rejected = 0
    for path in sorted(args.files):
        content = path.read_text(encoding="utf-8-sig")
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in content.split("# End of YAML header", 1)[1].splitlines():
            row = line.split()
            if not row:
                continue
            if len(row) != 16 or row[1:3] != ["C", "E"]:
                raise ValueError("只接受 C 星 ECEF 的 16 列 GNV1B 记录")
            t = int(row[0])
            if t % args.step:
                continue
            if int(row[15], 2) != 0:
                rejected += 1
                continue
            state = [t, *map(float, row[3:6]), *map(float, row[9:12])]
            if t in records and records[t] != state:
                raise ValueError("发现同历元不同值，不自动合并")
            records[t] = state
    data = np.array([records[t] for t in sorted(records)])
    if not np.isfinite(data).all():
        raise ValueError("存在无效数值")
    out = args.output
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        np.savetxt(stream, data, delimiter=",", fmt="%.17g", comments="",
            header="gps_seconds,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s")
    manifest = {"object": "GRACE-FO C", "product": "GNV1B RL04", "product_doi": "10.5067/GFJPL-L1B04",
        "frame": "Earth-fixed ECEF", "time": "GPS seconds since 2000-01-01 12:00:00 GPS; not UTC or receiver time",
        "source_files": sources, "selection": f"Exact epochs divisible by {args.step}, quality flag == 0; no interpolation or fitting",
        "step_seconds": args.step,
        "sampled_nonzero_quality_removed": rejected, "rows": len(data),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(data)} epochs, rejected {rejected} flagged samples")


if __name__ == "__main__":
    main()
