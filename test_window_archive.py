"""用归档坐标重新评分，并检查 CPU/GPU、时间边界、源码及测速一致性。"""
import csv
import hashlib
import json
import unittest
from unittest.mock import patch
import numpy as np
from main import ROOT
from short_2019 import DATA_PATH
from no_warmup import FIT_HOURS, CONTEXT_HOURS, history, future_reference, prepare

ARCHIVE = ROOT/'reference/window-sweep'


class WindowArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)

    def test_scores_recomputed_from_positions_and_all_windows_present(self):
        for device in ['gpu']:
            folder = ARCHIVE/device
            prediction = np.genfromtxt(folder/'prediction.csv', delimiter=',', names=True)
            baseline = np.genfromtxt(folder/'baseline.csv', delimiter=',', names=True)
            self.assertEqual(len(prediction), 3*5*8*60)
            self.assertEqual(len(baseline), 3*8*60)
            with (folder/'summary.csv').open() as stream:
                scores = {(int(r['fit_hours']), int(r['context_hours'])):r for r in csv.DictReader(stream)}
            for hours in FIT_HOURS:
                for context in [0, *CONTEXT_HOURS]:
                    table = baseline if context==0 else prediction
                    errors = []
                    for n in range(1, 9):
                        selected = table[(table['fit_hours']==hours) & (table['context_hours']==context)
                                         & (table['window']==n)]
                        expected = future_reference(self.data, n)
                        np.testing.assert_array_equal(selected['gps_seconds'], expected[:, 0])
                        xyz = np.column_stack([selected[f'predicted_{c}_m'] for c in 'xyz'])
                        reference = np.column_stack([selected[f'reference_{c}_m'] for c in 'xyz'])
                        np.testing.assert_array_equal(reference, expected[:, 1:4])
                        error = np.linalg.norm(xyz-reference, axis=1)
                        np.testing.assert_allclose(error, selected['error_3d_m'], rtol=0, atol=1e-9)
                        errors.append(error)
                    errors = np.array(errors)
                    row = scores[hours, context]
                    self.assertAlmostEqual(float(row['pooled_rms_3d_m']), np.sqrt(np.mean(errors**2)), places=9)
                    self.assertAlmostEqual(float(row['endpoint_rmse_3d_m']), np.sqrt(np.mean(errors[:, -1]**2)), places=9)

    def test_fit_boundaries_history_and_reusable_dynamics(self):
        table = np.genfromtxt(ARCHIVE/'gpu/baseline.csv', delimiter=',', names=True)
        for hours in FIT_HOURS:
            for n in range(1, 9):
                past = history(self.data, n, hours)
                folder = ARCHIVE/'gpu'/f'fit-{hours:02d}h-window-{n:02d}'
                info = json.loads((folder/'fit.json').read_text())
                self.assertEqual(info['fit_start_gps_seconds'], past[0, 0])
                self.assertEqual(info['fit_end_gps_seconds'], past[-1, 0])
                self.assertEqual(info['fit_points'], len(past))
                with patch('dynamics_cache.fit_and_predict', side_effect=AssertionError('缓存应可复用')):
                    arc = prepare(past)
                with np.load(folder/'history.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['gps_seconds'], past[::6, 0])
                    np.testing.assert_array_equal(saved['residual_swr'], arc['residual'])
                selected = table[(table['fit_hours']==hours) & (table['window']==n)]
                expected = np.column_stack([selected[f'predicted_{c}_m'] for c in 'xyz'])
                np.testing.assert_array_equal(arc['baseline'], expected)

    def test_source_fingerprints_device_agreement_and_benchmark(self):
        for folder in [ARCHIVE/'gpu', ARCHIVE/'benchmark/cpu', ARCHIVE/'benchmark/gpu']:
            env = json.loads((folder/'environment.json').read_text())
            for name, digest in env['source_sha256'].items():
                self.assertEqual(hashlib.sha256((ARCHIVE/'source'/name).read_bytes()).hexdigest(), digest)
        hashes = []
        for device in ['cpu', 'gpu']:
            folder = ARCHIVE/'benchmark'/device
            env = json.loads((folder/'environment.json').read_text())
            hashes.append(env['input_sha256'])
            with (folder/'timing.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertGreaterEqual(env['repeats'], 3)
            for row in rows:
                times = [float(row[f'repeat_{i+1}_seconds']) for i in range(env['repeats'])]
                self.assertGreater(min(times), 0)
                self.assertAlmostEqual(float(row['median_seconds']), float(np.median(times)), places=9)
        self.assertEqual(hashes[0], hashes[1])
        with np.load(ARCHIVE/'benchmark/cpu/forecasts.npz') as c, np.load(ARCHIVE/'benchmark/gpu/forecasts.npz') as g:
            for key in c.files:
                np.testing.assert_allclose(c[key], g[key], rtol=0, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
