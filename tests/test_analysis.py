import unittest

from quantum_seeker_v2 import QuantumSeekerFramework


class TestAnalysisPipeline(unittest.TestCase):
    def test_execute_full_analysis(self) -> None:
        config = {
            "random_seed": 2,
            "num_legs": 250,
            "max_parlays_per_size": 50,
            "max_parlays_analyzed": 30,
            "monte_carlo_sims": 50,
        }
        framework = QuantumSeekerFramework(config=config)
        results = framework.execute_full_analysis()

        self.assertIn("metadata", results)
        self.assertIn("quantum_needles", results)
        self.assertIn("golden_needles", results)
        self.assertIn("temporal", results)
        self.assertIn("inefficiencies", results)

        metadata = results["metadata"]
        self.assertGreaterEqual(metadata["parlays_analyzed"], len(results.get("top_constructions", [])))


if __name__ == "__main__":
    unittest.main()
