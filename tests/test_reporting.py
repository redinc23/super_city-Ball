import unittest

from quantum_seeker_v2 import QuantumSeekerFramework


class TestReporting(unittest.TestCase):
    def test_report_generation(self) -> None:
        config = {
            "random_seed": 3,
            "num_legs": 200,
            "max_parlays_analyzed": 20,
            "monte_carlo_sims": 20,
        }
        framework = QuantumSeekerFramework(config=config)
        framework.execute_full_analysis()
        report = framework.generate_full_report()

        self.assertIn("EXECUTIVE SUMMARY", report)
        self.assertIn("ANALYSIS COMPLETE", report)

        html_report = framework.generate_html_report(report_text=report)
        self.assertIn("<html>", html_report)
        self.assertIn("Quantum Seeker 2.0 Report", html_report)


if __name__ == "__main__":
    unittest.main()
