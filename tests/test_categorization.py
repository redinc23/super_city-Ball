import sys
from unittest.mock import MagicMock

# Mock numpy and pandas before they are imported by quantum_seeker_v2
mock_np = MagicMock()
mock_pd = MagicMock()

# Setup minimal mocks to allow import and basic usage
mock_np.random.default_rng.return_value = MagicMock()

sys.modules["numpy"] = mock_np
sys.modules["pandas"] = mock_pd
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.linear_model"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["seaborn"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()

import unittest
from quantum_seeker_v2 import QuantumSeekerFramework, ULTRA_RARE_BET_CATEGORIES, CATEGORY_PROFILES

class TestCategorization(unittest.TestCase):
    def setUp(self):
        # We use __new__ and only set what's necessary to avoid running __init__
        # which depends heavily on numpy/pandas even with mocks.
        self.framework = QuantumSeekerFramework.__new__(QuantumSeekerFramework)

    def test_categorize_all_known_bets(self):
        """Verify that every bet in ULTRA_RARE_BET_CATEGORIES is correctly categorized."""
        for expected_cat, bets in ULTRA_RARE_BET_CATEGORIES.items():
            for bet_type in bets:
                actual_cat = self.framework._categorize(bet_type)
                self.assertEqual(
                    actual_cat,
                    expected_cat,
                    f"Bet type '{bet_type}' should be in category '{expected_cat}', but got '{actual_cat}'"
                )

    def test_categorize_unknown_bet(self):
        """Verify that unknown bet types default to 'TRADITIONAL'."""
        self.assertEqual(self.framework._categorize("UNKNOWN_BET_TYPE"), "TRADITIONAL")

    def test_all_categories_have_profiles(self):
        """Verify that every category defined in ULTRA_RARE_BET_CATEGORIES has an entry in CATEGORY_PROFILES."""
        for cat in ULTRA_RARE_BET_CATEGORIES.keys():
            self.assertIn(
                cat,
                CATEGORY_PROFILES,
                f"Category '{cat}' found in ULTRA_RARE_BET_CATEGORIES but missing in CATEGORY_PROFILES"
            )

if __name__ == "__main__":
    unittest.main()
