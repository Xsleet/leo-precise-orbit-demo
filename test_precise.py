"""重点核验单位、潮汐相位、时标及不使用未来空间天气。无需下载 TimesFM。"""
import hashlib
import json
import unittest
import numpy as np
from precise import engine, PastWeather, DATA, Arc
from main import ROOT, select_window


class PreciseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = engine()

    def test_force_data_hashes_and_degree(self):
        for name, record in json.loads((DATA / 'manifest.json').read_text()).items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), record['sha256'])
        self.assertEqual(self.e['gravity'].getMaxDegree(), 150)
        self.assertEqual(self.e['gravity'].getMaxOrder(), 150)

    def test_gps_utc_and_precise_frame_round_trip(self):
        from org.orekit.utils import PVCoordinates
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        data = np.loadtxt(ROOT / 'data/orbit.csv', delimiter=',', skiprows=1)
        row = select_window(data, 1)[500]
        date = self.e['epoch'].shiftedBy(float(row[0]))
        utc = date.getComponents(self.e['utc'])
        self.assertEqual(str(utc.getDate()), '2025-06-01')
        self.assertAlmostEqual(utc.getTime().getSecondsInLocalDay(), 86400 - 18)
        pv = PVCoordinates(Vector3D(*row[1:4]), Vector3D(*row[4:]))
        transform = self.e['itrf'].getTransformTo(self.e['inertial'], date)
        back = transform.getInverse().transformPVCoordinates(transform.transformPVCoordinates(pv))
        np.testing.assert_allclose(back.getPosition().toArray(), row[1:4], atol=1e-8, rtol=0)
        np.testing.assert_allclose(back.getVelocity().toArray(), row[4:], atol=1e-10, rtol=0)

    def test_future_weather_does_not_change_forecast_input(self):
        from org.orekit.time import AbsoluteDate
        cutoff = AbsoluteDate(2025, 6, 2, 5, 0, 0., self.e['utc'])
        weather = PastWeather(cutoff)
        values = [weather.getInstantFlux(cutoff), weather.getMeanFlux(cutoff),
                  weather.getThreeHourlyKP(cutoff), weather.get24HoursKp(cutoff)]
        day = int(weather.cutoff // 86400)
        bin_ = int(weather.cutoff // 10800)
        weather.flux[day:] = 9999
        weather.mean[day:] = 9999
        weather.kp[bin_:] = 9999
        future = cutoff.shiftedBy(3600.)
        self.assertEqual(values, [weather.getInstantFlux(future), weather.getMeanFlux(future),
                                 weather.getThreeHourlyKP(future), weather.get24HoursKp(future)])

    def test_native_tides_against_original_gfc_coefficients(self):
        import jpype
        from org.orekit.forces.gravity.potential import GravityFieldFactory
        from org.orekit.utils import IERSConventions
        raw = json.loads((DATA / 'eot11a-check.json').read_text())
        waves = {w.getDoodson(): w for w in GravityFieldFactory.getOceanTidesWaves(120, 120)}
        self.assertEqual(len(waves), 18)
        date = self.e['epoch'].shiftedBy(802094400.)
        el = IERSConventions.IERS_2010.getNutationArguments(self.e['ut1']).evaluateAll(date)
        # Doodson tau, s, h, p, N', ps 从 Delaunay 基本角独立组成。
        s = el.getF() + el.getOmega()
        h = s - el.getD()
        fundamentals = np.array([el.getGamma() - s, s, h, s - el.getL(), -el.getOmega(), h - el.getLPrime()])
        for name, doodson in [('OM1', 55565), ('K1', 165555), ('M2', 255555)]:
            digits = np.array([int(c) for c in f'{doodson:06d}']) - np.array([0,5,5,5,5,5])
            phase = digits @ fundamentals
            c = jpype.JArray(jpype.JDouble, 2)(np.zeros((121,121)).tolist())
            s_ = jpype.JArray(jpype.JDouble, 2)(np.zeros((121,121)).tolist())
            waves[doodson].addContribution(el, c, s_)
            for key in raw[name]['cos']:
                n,m = map(int,key.split(','))
                expected = np.array(raw[name]['cos'][key])*np.cos(phase) + np.array(raw[name]['sin'][key])*np.sin(phase)
                np.testing.assert_allclose([c[n][m],s_[n][m]], expected, rtol=1e-9, atol=1e-22)

    def test_force_stack_and_short_propagation(self):
        data = np.loadtxt(ROOT / 'data/orbit.csv', delimiter=',', skiprows=1)
        past = select_window(data, 1)[:11]
        arc = Arc(past)
        names = [str(f.getClass().getSimpleName()) for f in arc.forces]
        for required in ['SolidTides','OceanTides','DragForce','SolarRadiationPressure','Relativity']:
            self.assertIn(required, names)
        self.assertEqual(names.count('ThirdBodyAttraction'), 2)
        self.assertEqual(names.count('ParametricAcceleration'), 4)
        states = arc.propagate(np.zeros(len(arc.scale)), np.arange(11)*60., earth_fixed=True)
        np.testing.assert_allclose(states[0], past[0,1:], atol=1e-8, rtol=0)
        self.assertTrue(np.isfinite(states).all())
        self.assertTrue(np.all((np.linalg.norm(states[:,:3],axis=1)>6.7e6)&(np.linalg.norm(states[:,:3],axis=1)<7e6)))

    def test_final_drag_segment_has_full_orbit_of_observations(self):
        data = np.loadtxt(ROOT/'data/orbit.csv', delimiter=',', skiprows=1)
        arc = Arc(select_window(data, 1)[:481])
        self.assertGreater(arc.end.durationFrom(arc.drag_dates[-1]), arc.period - .01)
        self.assertEqual(len(arc.drag_dates), 5)

    def test_variational_derivatives_against_parameter_perturbation(self):
        data = np.loadtxt(ROOT/'data/orbit.csv', delimiter=',', skiprows=1)
        arc = Arc(select_window(data, 1)[:61])
        p = np.zeros(len(arc.scale)); seconds = np.arange(61)*60.
        states, jacobian = arc.propagate(p, seconds, with_jacobian=True)
        for index in [0, 6, len(p)-1]:
            changed = arc.propagate(p + np.eye(len(p))[index], seconds)
            difference = (changed[:,:3] - states[:,:3]).ravel()
            relative = np.linalg.norm(difference - jacobian[:,index]) / np.linalg.norm(jacobian[:,index])
            self.assertLess(relative, .003)


class PreciseArchiveTests(unittest.TestCase):
    def test_all_windows_reuse_matching_cache_and_scores_recompute(self):
        import csv
        from unittest.mock import patch
        from main import predict, HISTORY
        from compare_windows import summarize
        data = np.loadtxt(ROOT / 'data/orbit.csv', delimiter=',', skiprows=1)
        for device in ['cpu', 'gpu']:
            folder = ROOT / 'reference/precise' / device
            errors = {}
            for number in range(1,9):
                sample = select_window(data, number)
                pred = np.genfromtxt(folder / f'window-{number:02d}/prediction.csv', delimiter=',', names=True)
                run = json.loads((folder / f'window-{number:02d}/run.json').read_text())
                np.testing.assert_array_equal(pred['gps_seconds'], sample[HISTORY:,0])
                np.testing.assert_array_equal(np.column_stack([pred[f'reference_{c}_m'] for c in 'xyz']), sample[HISTORY:,1:4])
                self.assertEqual(run['fit']['fit_end_gps_seconds'], sample[500,0]-1200)
                self.assertEqual(run['fresh_fit']['fit_end_gps_seconds'], sample[500,0])
                for name in ['rdop','fresh_rdop','rdop_linear','rdop_timesfm']:
                    positions = np.column_stack([pred[f'{name}_{c}_m'] for c in 'xyz'])
                    error = np.linalg.norm(positions - sample[HISTORY:,1:4],axis=1)
                    np.testing.assert_allclose(error,pred[f'{name}_error_3d_m'],atol=1e-8,rtol=0)
                    errors.setdefault(name,[]).append(error)
                if device == 'cpu':
                    with patch('dynamics_cache.fit_and_predict',side_effect=AssertionError('附带动力学缓存失效')):
                        result, details = predict(sample[:HISTORY])
                    for name in result:
                        np.testing.assert_allclose(result[name],np.column_stack([pred[f'{name}_{c}_m'] for c in 'xyz']),atol=2e-8,rtol=0)
                    for key in ['fit','fresh_fit']:
                        self.assertTrue((ROOT/'reference/precise/dynamics-cache'/f"{details[key]['cache_key']}.npz").exists())
            summaries,_ = summarize(errors)
            with (folder/'summary.csv').open(encoding='utf8') as stream:
                actual = {r['model']: r for r in csv.DictReader(stream)}
            for row in summaries:
                for key in ['pooled_rms_3d_m','endpoint_rmse_3d_m']:
                    self.assertAlmostEqual(row[key],float(actual[row['model']][key]),places=8)

    def test_cached_orbit_reconstructs_from_past_fit_parameters(self):
        from main import FIT
        folder = ROOT/'reference/precise/gpu/window-01'
        info = json.loads((folder/'run.json').read_text())['fit']
        sample = select_window(np.loadtxt(ROOT/'data/orbit.csv',delimiter=',',skiprows=1),1)
        arc = Arc(sample[:FIT+1])
        trajectory = arc.propagate(np.array(info['parameters_scaled']),np.arange(561)*60.,earth_fixed=True)
        pred = np.genfromtxt(folder/'prediction.csv',delimiter=',',names=True)
        positions = np.column_stack([pred[f'rdop_{c}_m'] for c in 'xyz'])
        self.assertLess(float(np.linalg.norm(trajectory[501:,:3]-positions,axis=1).max()),.005)


if __name__ == '__main__':
    unittest.main()
