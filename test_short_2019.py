"""验证换年份、10 秒拟合和 80 点输入时，没有混用日期或读取未来。"""
import csv
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from main import ROOT, HISTORY, predict
from short_2019 import DATA_PATH, FIRST, window


class Short2019Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)

    def test_data_hash_and_all_fixed_windows(self):
        manifest = json.loads(DATA_PATH.with_suffix('.manifest.json').read_text())
        self.assertEqual(hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(), manifest['output_sha256'])
        for n in range(1,9):
            sample, past = window(self.data, n)
            self.assertEqual(sample.shape, (561,7))
            self.assertEqual(past.shape, (3001,7))
            self.assertEqual(past[-1,0], FIRST+(n-1)*3600)
            np.testing.assert_array_equal(past[::6],sample[:HISTORY])
        with self.assertRaises(ValueError):
            window(np.delete(self.data, np.searchsorted(self.data[:,0],FIRST), axis=0), 1)

    def test_only_past_10second_data_reaches_fit_and_80_residuals_reach_model(self):
        sample, past = window(self.data,1)
        seen, contexts = [], []
        def fit(observed, future_points, **kwargs):
            seen.append(observed.copy())
            count = int((observed[-1,0]-observed[0,0])/60)+1+future_points
            return np.tile(observed[0,1:], (count,1)), {}
        class Evaluator:
            def predict_batch(self, inputs, **kwargs):
                contexts.append(np.array(inputs))
                return [SimpleNamespace(forecast=np.zeros(60)) for _ in range(3)]
        with patch('dynamics_cache.fit_cached', side_effect=fit):
            result, details = predict(sample[:HISTORY],Evaluator(),fit_history=past,context_points=80)
        self.assertEqual([len(x) for x in seen], [2881,2881])
        self.assertEqual(seen[0][-1,0], FIRST-1200)
        self.assertEqual(seen[1][-1,0], FIRST)
        self.assertEqual(contexts[0].shape, (3,80))
        self.assertEqual(result['rdop'].shape,(60,3))
        error = details['past_error_swr']
        np.testing.assert_allclose(contexts[0],(error[-80:]-error[-1]).T,rtol=1e-6,atol=1e-3)
        bad = past.copy(); bad[-1,0] += 10
        with self.assertRaises(ValueError):
            predict(sample[:HISTORY],fit_history=bad)

    def test_weather_reads_2019_and_freezes_before_future(self):
        from precise import engine, PastWeather
        e = engine()
        from org.orekit.time import AbsoluteDate
        cutoff = AbsoluteDate(2019,1,2,0,0,0.,e['utc'])
        weather = PastWeather(cutoff)
        self.assertAlmostEqual(weather.getInstantFlux(cutoff),71.9)
        self.assertAlmostEqual(weather.getThreeHourlyKP(cutoff),.7)
        self.assertAlmostEqual(weather.get24HoursKp(cutoff),1.2125)
        expected = [weather.getInstantFlux(cutoff),weather.getThreeHourlyKP(cutoff)]
        day = int(weather.elapsed(cutoff)//86400)
        weather.flux[day:] = 9999
        weather.kp[day*8:] = 9999
        future = cutoff.shiftedBy(3600.)
        self.assertEqual(expected,[weather.getInstantFlux(future),weather.getThreeHourlyKP(future)])


class Short2019ArchiveTests(unittest.TestCase):
    def test_full_results_scores_and_source_snapshots(self):
        from compare_windows import summarize
        data = np.loadtxt(DATA_PATH,delimiter=',',skiprows=1)
        archive = ROOT/'reference/2019-short'
        for device in ['cpu','gpu']:
            folder = archive/device
            metadata = json.loads((folder/'environment.json').read_text())
            self.assertEqual(metadata['data_sha256'],hashlib.sha256(DATA_PATH.read_bytes()).hexdigest())
            for name,sha in metadata['source_sha256'].items():
                self.assertEqual(hashlib.sha256((archive/'source'/name).read_bytes()).hexdigest(),sha)
            errors = {}
            for number in range(1,9):
                sample,past = window(data,number)
                run = json.loads((folder/f'window-{number:02d}/run.json').read_text())
                pred = np.genfromtxt(folder/f'window-{number:02d}/prediction.csv',delimiter=',',names=True)
                np.testing.assert_array_equal(pred['gps_seconds'],sample[HISTORY:,0])
                self.assertEqual(run['context_points'],80)
                self.assertEqual(run['fit']['fit_points'],2881)
                self.assertEqual(run['fit']['fit_end_gps_seconds'],past[-1,0]-1200)
                self.assertEqual(run['fresh_fit']['fit_end_gps_seconds'],past[-1,0])
                for name in ['rdop','rdop_linear','fresh_rdop','rdop_timesfm']:
                    xyz = np.column_stack([pred[f'{name}_{c}_m'] for c in 'xyz'])
                    error = np.linalg.norm(xyz-sample[HISTORY:,1:4],axis=1)
                    np.testing.assert_allclose(error,pred[name+'_error_3d_m'],rtol=0,atol=1e-8)
                    errors.setdefault(name,[]).append(error)
                if device=='cpu':
                    with patch('dynamics_cache.fit_and_predict',side_effect=AssertionError('2019 缓存不匹配')):
                        rebuilt,_ = predict(sample[:HISTORY],fit_history=past,context_points=80)
                    for name,xyz in rebuilt.items():
                        expected = np.column_stack([pred[f'{name}_{c}_m'] for c in 'xyz'])
                        np.testing.assert_allclose(xyz,expected,rtol=0,atol=2e-8)
            summary,_ = summarize(errors)
            with (folder/'summary.csv').open() as stream:
                stored = {row['model']:row for row in csv.DictReader(stream)}
            for row in summary:
                for key in ['pooled_rms_3d_m','endpoint_rmse_3d_m']:
                    self.assertAlmostEqual(row[key],float(stored[row['model']][key]),places=8)


if __name__ == '__main__':
    unittest.main()
