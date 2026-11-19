import unittest
import pandas as pd

from src.drift_monitor import psi


class TestPSI(unittest.TestCase):
    def test_identical_distributions_have_near_zero_psi(self):
        values = pd.Series(range(100), dtype=float)
        self.assertAlmostEqual(psi(values, values), 0.0, places=8)

    def test_psi_is_non_negative(self):
        expected = pd.Series(range(100), dtype=float)
        actual = pd.Series(range(50, 150), dtype=float)
        self.assertGreaterEqual(psi(expected, actual), 0.0)


if __name__ == "__main__":
    unittest.main()
