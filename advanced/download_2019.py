"""从 GFZ 官方镜像重建 2019 短期实验数据（两个压缩包共约 317 MB）。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://isdc-data.gfz.de/grace-fo/Level-1B/JPL/INSTRUMENT/RL04/2019/'


def main():
    raw = ROOT/'data/raw/2019'
    raw.mkdir(parents=True, exist_ok=True)
    sources, files = [], []
    for day in [1,2]:
        name = f'gracefo_1B_2019-01-{day:02d}_RL04.ascii.noLRI.tgz'
        archive_path = raw/name
        if not archive_path.exists():
            print('下载',name,flush=True)
            with urlopen(BASE+name,timeout=180) as response:
                archive_path.write_bytes(response.read())
        filename = f'GNV1B_2019-01-{day:02d}_C_04.txt'
        target = raw/filename
        with tarfile.open(archive_path) as archive:
            member = next(m for m in archive.getmembers() if Path(m.name).name==filename)
            target.write_bytes(archive.extractfile(member).read())
        files.append(str(target))
        sources.append(dict(url=BASE+name,archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                            file=filename,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    expected = ROOT/'data/orbit-2019.sources.json'
    if expected.exists() and json.loads(expected.read_text()) != sources:
        raise ValueError('下载来源的哈希与已归档实验不同，停止重建')
    expected.write_text(json.dumps(sources,indent=2)+'\n',encoding='utf8')
    subprocess.run([sys.executable,str(ROOT/'advanced/prepare_data.py'),*files,
                    '--step','10','--output',str(ROOT/'data/orbit-2019.csv')],check=True)


if __name__=='__main__':
    main()
