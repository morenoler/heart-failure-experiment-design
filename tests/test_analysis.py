import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ab_plan import sample_size_per_arm, two_proportion_test
from analyze import bootstrap_interval, load_data, permutation_test


class AnalysisTests(unittest.TestCase):
    def test_source_checks(self):
        frame = load_data()
        self.assertEqual(len(frame), 299)
        self.assertEqual(frame["DEATH_EVENT"].sum(), 96)

    def test_permutation_is_reproducible(self):
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([4.0, 5.0, 6.0])
        self.assertEqual(permutation_test(a, b, 100, 7), permutation_test(a, b, 100, 7))

    def test_bootstrap_interval_on_constants(self):
        self.assertEqual(bootstrap_interval(np.ones(3), np.zeros(4), 100, 7), (1.0, 1.0))

    def test_ab_counts_and_plan(self):
        self.assertGreater(sample_size_per_arm(0.60, 0.08), 0)
        result = two_proportion_test(600, 1000, 680, 1000)
        self.assertAlmostEqual(result["difference"], 0.08)
        self.assertLess(result["p_value_two_sided"], 0.05)


if __name__ == "__main__":
    unittest.main()
