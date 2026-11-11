"""This tree's results on the shared scoring scenarios (tests/scoring_scenarios.py).

The expected passes below are origin/main's (bb42bd5) results, except where main is the defect: there,
the fix must hold back main's tick or recover the answer main loses. scripts/scoring_differential.py
checks the same scenarios against another tree directly.
"""

import asyncio
import unittest

from callcoach import app

from . import scoring_scenarios as sc

EXPECTED = {  # question -> the pass it first ticks on
    # From the PR 13 review, round 2 (regressions against main, now matching main):
    "continued_speech_window": {"1": 2},
    "fit_failed_asked_during_window": {"2": 1},
    "stretch_interleave_reask": {"1": 1},
    # Parity with main:
    "normal_two_passes": {"1": 2},
    "failed_line_then_new_evidence": {"2": 3},
    "failed_fit_window": {"1": 2},
    "reask_during_stretch_wait": {"1": 3},
    "answer_before_question": {"2": 2},
    # The round-4 defects in main: main ticks 2 and 1 here wrongly, and never ticks 1 in the last.
    "auto_during_partial_retry": {},
    "evidence_counted_twice": {},
    "evidence_closes_back": {"1": 2},
}


class ScenarioTests(unittest.TestCase):
    def tearDown(self):
        app.jev = None

    def test_every_scenario_is_expected(self):
        self.assertEqual(set(EXPECTED), set(sc.SCENARIOS))

    def test_each_scenario(self):
        for name, scenario in sc.SCENARIOS.items():
            with self.subTest(name):
                try:
                    self.assertEqual(asyncio.run(scenario(app)), EXPECTED[name])
                finally:
                    app.jev = None


if __name__ == "__main__":
    unittest.main()
