"""核对首页精度表对应的统一评分及动力学缓存。"""
import csv
import unittest
from unittest.mock import patch
import numpy as np
from main import ROOT
from no_warmup import FIT_HOURS, CONTEXT_HOURS, history, future_reference, prepare
from short_2019 import DATA_PATH

REFERENCE = ROOT / 'reference'


def read_rows(name):
    with (REFERENCE / name).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


class OrbitAccuracyTests(unittest.TestCase):
    def test_all_combinations_and_rmse(self):
        expected = {(h, c) for h in FIT_HOURS for c in (0, *CONTEXT_HOURS) if c <= h}
        summary = read_rows('summary.csv')
        windows = read_rows('by_window.csv')
        leads = read_rows('by_lead.csv')
        self.assertEqual(len(summary), len(expected))
        self.assertEqual(len(windows), len(expected) * 8)
        self.assertEqual(len(leads), len(expected) * 60)
        self.assertEqual({(int(row['fit_hours']), int(row['context_hours'])) for row in summary}, expected)
        for row in summary:
            fit, context = row['fit_hours'], row['context_hours']
            per_window = [r for r in windows if (r['fit_hours'], r['context_hours']) == (fit, context)]
            per_lead = [r for r in leads if r['model'] == row['model']]
            self.assertEqual([int(r['window']) for r in per_window], list(range(1, 9)))
            self.assertEqual([int(r['lead_minutes']) for r in per_lead], list(range(1, 61)))
            endpoint = np.sqrt(np.mean([float(r['endpoint_3d_m'])**2 for r in per_window]))
            pooled = np.sqrt(np.mean([float(r['rms_3d_m'])**2 for r in per_window]))
            self.assertAlmostEqual(endpoint, float(row['endpoint_rmse_3d_m']), places=9)
            self.assertAlmostEqual(pooled, float(row['pooled_rms_3d_m']), places=9)
            self.assertAlmostEqual(endpoint, float(per_lead[-1]['rmse_3d_m']), places=9)
            self.assertAlmostEqual(pooled, np.sqrt(np.mean([float(r['rmse_3d_m'])**2 for r in per_lead])), places=9)

    def test_baseline_scores_from_cached_coordinates(self):
        data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
        windows = {(int(r['fit_hours']), int(r['window'])): r for r in read_rows('by_window.csv')
                   if r['context_hours'] == '0'}
        for hours in FIT_HOURS:
            for window in range(1, 9):
                past = history(data, window, hours)
                with patch('dynamics_cache.fit_and_predict', side_effect=AssertionError('应复用动力学缓存')):
                    arc = prepare(past)
                info = arc['fit']
                self.assertEqual(info['fit_start_gps_seconds'], past[0, 0])
                self.assertEqual(info['fit_end_gps_seconds'], past[-1, 0])
                self.assertEqual(info['fit_points'], len(past))
                reference = future_reference(data, window)
                errors = np.linalg.norm(arc['baseline'] - reference[:, 1:4], axis=1)
                row = windows[hours, window]
                self.assertAlmostEqual(float(row['endpoint_3d_m']), errors[-1], places=9)
                self.assertAlmostEqual(float(row['rms_3d_m']), np.sqrt(np.mean(errors**2)), places=9)


if __name__ == '__main__':
    unittest.main()
