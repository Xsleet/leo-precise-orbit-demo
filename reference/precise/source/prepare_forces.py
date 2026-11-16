"""核验已附带的力模型；加 --download 可从 manifest 中的固定来源重建。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/forces'


def get(url):
    with urlopen(url, timeout=180) as response:
        return response.read()


def rebuild(name, record):
    if name == 'EGM2008-150.gfc':
        lines = []
        with urlopen(record['url'], timeout=180) as response:
            for raw in response:
                line = raw.rstrip(b'\r\n')
                if line.startswith(b'gfc') and int(line.split()[1]) > 150:
                    break
                lines.append(b'max_degree 150' if line.startswith(b'max_degree') else line)
        return b'\n'.join(lines) + b'\n'
    if name != 'eot11a-potential.dat':
        return get(record['url'])
    sources = record['sources']
    def fetch(item):
        key, source = item
        raw = get(source['url'])
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError('上游 EOT11a 文件发生变化：' + key)
        rows = [[float(v) for v in line.split()[1:5]] for line in raw.decode().splitlines() if line.startswith('gfc')]
        return key, rows
    with ThreadPoolExecutor(max_workers=6) as pool:
        arrays = dict(pool.map(fetch, sources.items()))
    waves = dict(zip(['OM1','OM2','SA','SSA','MM','MF','MTM','MSQ','Q1','O1','P1','K1','2N2','N2','M2','S2','K2','M4'],
                    ['055.565','055.575','056.554','057.555','065.455','075.555','086.455','093.555',
                     '135.655','145.555','163.655','165.555','235.755','245.655','255.555','273.555','275.555','455.555']))
    lines = ['EOT11a 18 main waves, dimensionless fully normalized potential coefficients incl loading and phase']
    for wave, doodson in waves.items():
        for (n,m,cc,sc),(_,_,cs,ss) in zip(arrays[f'eot11a.{wave}.cos.gfc'], arrays[f'eot11a.{wave}.sin.gfc']):
            if n >= 2:
                lines.append(f'{doodson} {wave} {int(n)} {int(m)} {(cc-ss)/2:.15e} {(cs+sc)/2:.15e} {(cc+ss)/2:.15e} {(cs-sc)/2:.15e}')
    return ('\n'.join(lines) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name, record in manifest.items():
        raw = rebuild(name, record) if args.download else (DATA / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('SHA-256 不一致：' + name)
        if args.download:
            (DATA / name).write_bytes(raw)
        print('OK', name)


if __name__ == '__main__':
    main()
