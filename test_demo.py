"""无需下载 TimesFM：验证动力学、数据和时间边界。"""
import hashlib
import json
from types import SimpleNamespace
import unittest

import numpy as np
import main as demo
from orbit import MU, R_EARTH, J2, OMEGA, gravity, propagate


class PhysicsTests(unittest.TestCase):
    def test_circular_orbit_against_analytic_solution(self):
        radius = 7e6
        speed = np.sqrt(MU / radius)
        states = propagate([radius, 0, 0, 0, speed, 0], use_j2=False)
        phase = np.sqrt(MU / radius**3) * np.arange(1, 61) * 60
        exact = radius * np.column_stack([np.cos(phase), np.sin(phase), np.zeros(60)])
        self.assertLess(np.max(np.linalg.norm(states[:, :3] - exact, axis=1)), .02)
        energy = np.sum(states[:, 3:]**2, axis=1) / 2 - MU / np.linalg.norm(states[:, :3], axis=1)
        self.assertLess(np.max(np.abs(energy / (-MU / (2 * radius)) - 1)), 1e-8)

    def test_j2_matches_potential_gradient(self):
        position = np.array([5e6, 3e6, 4e6])
        def potential(r):
            radius = np.linalg.norm(r)
            return -MU / radius * (1 - J2 * (R_EARTH / radius)**2 * (.5 * (3 * (r[2] / radius)**2 - 1)))
        gradient = np.array([(potential(position + d) - potential(position - d)) / 2
                             for d in np.eye(3)])
        np.testing.assert_allclose(gravity(position), -gradient, atol=3e-8, rtol=0)

    def test_rotating_frame_matches_inertial_frame(self):
        initial = np.array([7e6, 0, 0, 0, 6500, 3800.])
        inertial = propagate(initial)
        rotating_initial = initial.copy()
        rotating_initial[3:] -= np.cross(OMEGA, initial[:3])
        rotating = propagate(rotating_initial, rotating=True)
        angles = OMEGA[2] * np.arange(1, 61) * 60
        x, y, z = inertial[:, :3].T
        expected = np.column_stack([np.cos(angles)*x + np.sin(angles)*y,
                                    -np.sin(angles)*x + np.cos(angles)*y, z])
        self.assertLess(np.max(np.linalg.norm(rotating[:, :3] - expected, axis=1)), .02)
        finer = propagate(rotating_initial, rotating=True, max_step=2.5)
        self.assertLess(np.max(np.linalg.norm(rotating[:, :3] - finer[:, :3], axis=1)), .02)


    def test_j2_backward_arc_returns_to_origin(self):
        initial = np.array([7e6, 0, 0, 0, 6500, 3800.])
        back = propagate(initial, points=359, interval=-60, rotating=True)
        recovered = propagate(back[-1], points=359, interval=60, rotating=True)[-1]
        self.assertLess(np.linalg.norm(recovered[:3] - initial[:3]), .05)
        self.assertLess(np.linalg.norm(recovered[3:] - initial[3:]), .0001)
        with self.assertRaises(ValueError):
            propagate(initial, interval=0)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.loadtxt(demo.ROOT / "data/orbit.csv", delimiter=",", skiprows=1)

    def test_manifest_and_time_grid(self):
        manifest = json.loads((demo.ROOT / "data/orbit.manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["output_sha256"], hashlib.sha256((demo.ROOT / "data/orbit.csv").read_bytes()).hexdigest())
        for number in range(1, demo.WINDOWS + 1):
            sample = demo.select_window(self.data, number)
            self.assertEqual(len(sample), 561)
            self.assertEqual(sample[500, 0], demo.origin_seconds(number))
            self.assertEqual(sample[480, 0], demo.origin_seconds(number) - 1200)
            self.assertEqual(sample[501, 0], demo.origin_seconds(number) + 60)
        sample = demo.select_window(self.data, 1)
        with self.assertRaises(ValueError):
            demo.select_window(np.delete(sample, 200, axis=0), 1)

    def test_warmup_is_excluded_from_fit_and_correction_uses_same_arc(self):
        from unittest.mock import patch
        from rdop import to_xyz
        seen, calls = [], []
        sample = demo.select_window(self.data, 1)
        class Evaluator:
            def predict_batch(self, inputs, **kwargs):
                seen.append(np.array(inputs))
                return [SimpleNamespace(forecast=np.zeros(60)) for _ in range(3)]
        def fake_fit(observed, points, high_order=True):
            calls.append(observed.copy())
            return np.tile(observed[0, 1:7], (len(observed) + points, 1)), {}
        with patch("main.fit_and_predict", side_effect=fake_fit):
            result, details = demo.predict(sample[:demo.HISTORY], Evaluator(), gravity="egm2008")
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(calls[0], sample[:481])
        np.testing.assert_array_equal(calls[1], sample[20:501])
        residual = details["past_error_swr"]
        np.testing.assert_allclose(seen[0], (residual[-360:] - residual[-1]).T, rtol=1e-6, atol=1e-3)
        correction = to_xyz(np.tile(residual[-1], (60, 1)), details["basis"][501:])
        np.testing.assert_allclose(result["rdop_timesfm"] - result["rdop"], correction, atol=1e-8)
        with self.assertRaises(ValueError):
            demo.predict(sample, Evaluator())

    def test_lead_rmse_is_not_mean_distance(self):
        from compare_windows import summarize
        summary, leads = summarize({"model": [np.array([3., 4.]), np.array([0., 0.])]})
        self.assertAlmostEqual(summary[0]["endpoint_rmse_3d_m"], np.sqrt(8))
        self.assertAlmostEqual(summary[0]["pooled_rms_3d_m"], 2.5)
        self.assertAlmostEqual(leads[0]["rmse_3d_m"], np.sqrt(4.5))


class ReducedDynamicsTests(unittest.TestCase):
    def test_swr_round_trip_and_orientation(self):
        from rdop import frame, to_swr, to_xyz
        state = np.array([7e6, 0, 0, 0, -OMEGA[2]*7e6, 7500.])
        basis = frame(state)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(basis[:, 0], [0, 0, 1], atol=1e-14)
        errors = np.array([[1., 2., 3.]])
        np.testing.assert_allclose(to_xyz(to_swr(errors, basis[None]), basis[None]), errors)

    def test_harmonic_gravity_against_independent_potential(self):
        from scipy.special import lpmv, gammaln
        from gravity import acceleration, C, S, GM, RADIUS
        c, s = C[:13, :13].copy(), S[:13, :13].copy()
        def potential(r):
            radius, lon = np.linalg.norm(r), np.arctan2(r[1], r[0])
            total = 0.
            for n in range(13):
                for m in range(n + 1):
                    norm = np.sqrt((2*n+1)*(1 if m == 0 else 2) *
                                   np.exp(gammaln(n-m+1) - gammaln(n+m+1)))
                    p = (-1)**m * lpmv(m, n, r[2]/radius) * norm
                    total += (RADIUS/radius)**n * p * (c[n,m]*np.cos(m*lon)+s[n,m]*np.sin(m*lon))
            return GM/radius*total
        for r in [np.array([5e6,3e6,4e6]), np.array([-2e5,3e5,-6.9e6])]:
            gradient = np.array([(potential(r+d) - potential(r-d))/2 for d in np.eye(3)])
            np.testing.assert_allclose(acceleration(r, c, s), gradient, atol=5e-8, rtol=0)

    def test_fit_recovers_synthetic_acceleration_and_future(self):
        from rdop import integrate, fit_and_predict
        initial = np.array([7e6, 0, 0, 0, -OMEGA[2]*7e6, np.sqrt(MU/7e6)])
        empirical = np.array([-2e-6, 3e-6, -1e-6, 2e-6, 4e-6])
        truth = integrate(initial, empirical, np.arange(181)*60., high_order=False)
        observed = np.column_stack([np.arange(121)*60., truth[:121]])
        recovered, info = fit_and_predict(observed, 60, high_order=False)
        self.assertLess(np.max(np.linalg.norm(recovered[121:, :3]-truth[121:, :3], axis=1)), .02)
        self.assertLess(info["fit_rms_3d_m"], .002)


if __name__ == "__main__":
    unittest.main()
