import unittest

from quantum_seeker_v2 import QuantumSeekerFramework


class TestDataGeneration(unittest.TestCase):
    def test_data_generation_count_and_ranges(self) -> None:
        config = {
            "random_seed": 1,
            "num_legs": 200,
            "max_parlays_analyzed": 25,
            "monte_carlo_sims": 50,
        }
        framework = QuantumSeekerFramework(config=config)
        self.assertEqual(len(framework.bet_legs), config["num_legs"])

        for leg in framework.bet_legs[:50]:
            self.assertTrue(0.0 < leg.implied_prob < 1.0)
            self.assertTrue(leg.odds_decimal >= 1.01)
            self.assertTrue(0.0 <= leg.liquidity_score <= 1.0)
            self.assertTrue(leg.bookmaker_count >= 1)

    def test_year_range_inclusive_and_validated(self) -> None:
        config = {
            "random_seed": 7,
            "num_legs": 20,
            "year_start": 2024,
            "year_end": 2024,
            "max_parlays_analyzed": 5,
            "monte_carlo_sims": 5,
        }
        framework = QuantumSeekerFramework(config=config)
        self.assertTrue(all(leg.season_year == 2024 for leg in framework.bet_legs))

        with self.assertRaises(ValueError):
            QuantumSeekerFramework(
                config={
                    "num_legs": 5,
                    "year_start": 2025,
                    "year_end": 2024,
                    "monte_carlo_sims": 5,
                    "max_parlays_analyzed": 2,
                }
            )


if __name__ == "__main__":
    unittest.main()
