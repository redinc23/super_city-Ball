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


if __name__ == "__main__":
    unittest.main()
