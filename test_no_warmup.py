"""变窗口实验的关键约束：拟合截止起报点、输入长度准确、未来只用于评分。"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from short_2019 import DATA_PATH, FIRST
from no_warmup import FIT_HOURS, CONTEXT_HOURS, history, future_reference, prepare, correct


class NoWarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)

    def test_all_windows_and_missing_data(self):
        for hours in FIT_HOURS:
            for n in range(1, 9):
                past = history(self.data, n, hours)
                future = future_reference(self.data, n)
                origin = FIRST + (n-1)*3600
                self.assertEqual(past.shape, (hours*360+1, 7))
                self.assertEqual(past[0, 0], origin-hours*3600)
                self.assertEqual(past[-1, 0], origin)
                self.assertEqual(future.shape, (60, 7))
                self.assertEqual(future[0, 0], origin+60)
                self.assertEqual(future[-1, 0], origin+3600)
        with self.assertRaises(ValueError):
            history(self.data[1:], 1, 24)
        damaged = self.data.copy(); damaged[100, 2] = np.nan
        with self.assertRaises(ValueError):
            history(damaged, 1, 24)

    def test_future_not_needed_and_each_context_uses_its_exact_tail(self):
        seen, contexts = [], []
        def fit(past, count, **kwargs):
            seen.append(past.copy())
            trajectory = np.concatenate([past[::6, 1:], np.tile(past[-1, 1:], (count, 1))])
            trajectory[:, 0] += np.linspace(0, .1, len(trajectory))
            return trajectory, {}
        class Evaluator:
            def predict_batch(self, inputs, **kwargs):
                contexts.append(np.array(inputs))
                return [SimpleNamespace(forecast=np.zeros(60)) for _ in range(3)]
        past_only = self.data[self.data[:, 0] <= FIRST]
        with patch('no_warmup.fit_cached', side_effect=fit):
            arc = prepare(history(past_only, 1, 8))
        self.assertEqual(seen[0][-1, 0], FIRST)
        self.assertEqual(len(seen[0]), 2881)
        self.assertEqual(arc['residual'].shape, (481, 3))
        for hours in CONTEXT_HOURS:
            result = correct(arc, Evaluator(), hours)
            self.assertEqual(result.shape, (60, 3))
            self.assertEqual(contexts[-1].shape, (3, hours*60))
            expected = (arc['residual'][-hours*60:] - arc['residual'][-1]).T.astype(np.float32)
            np.testing.assert_array_equal(contexts[-1], expected)


if __name__ == '__main__':
    unittest.main()
