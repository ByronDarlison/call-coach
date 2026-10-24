import unittest
from datetime import datetime

from callcoach.audio import _clean
from callcoach.jev import apply_scores
from callcoach.questions import Followup, parse_questions, render_results

SAMPLE = """# Interview

Intro paragraph that is not a question.

## Start
1. When did you start?
2) Who was first?

## Sales
- [ ] How many emails?
* What reply rate?
"""


class ParseTests(unittest.TestCase):
    def test_every_list_item_is_a_question_and_headings_are_sections(self):
        qs = parse_questions(SAMPLE)
        self.assertEqual([q.text for q in qs],
                         ["When did you start?", "Who was first?", "How many emails?", "What reply rate?"])
        self.assertEqual([q.section for q in qs], ["Start", "Start", "Sales", "Sales"])
        self.assertEqual([q.number for q in qs], [1, 2, 3, 4])


class ScoreTests(unittest.TestCase):
    def test_needs_two_high_passes_and_respects_manual(self):
        qs = parse_questions(SAMPLE)
        apply_scores(qs, {"1": (0.9, 0.1), "2": (0.95, 0.1)})
        self.assertFalse(qs[0].answered)
        qs[1].manual = True
        apply_scores(qs, {"1": (0.85, 0.1), "2": (0.95, 0.1)})
        self.assertTrue(qs[0].answered)
        self.assertFalse(qs[1].answered)

    def test_follow_ups_are_ticked_by_key_too(self):
        f = Followup("1.1", "Why March?", "mcp")
        apply_scores([f], {"1.1": (0.9, 0)})
        apply_scores([f], {"1.1": (0.9, 0)})
        self.assertTrue(f.answered)

    def test_a_low_pass_resets_the_count(self):
        qs = parse_questions(SAMPLE)
        apply_scores(qs, {"1": (0.9, 0)})
        apply_scores(qs, {"1": (0.2, 0)})
        apply_scores(qs, {"1": (0.9, 0)})
        self.assertFalse(qs[0].answered)


class OutputTests(unittest.TestCase):
    def test_results_nest_followups_and_keep_loose_ones_at_the_end(self):
        qs = parse_questions(SAMPLE)
        qs[0].answered = True
        qs[1].followups.append(Followup("2.1", "Why them?", "mcp", answered=True))
        loose = [Followup("f1", "Anything else?", "you"), Followup("f2", "What about pricing?", "mcp", True, True)]
        out = render_results("Interview", datetime(2026, 10, 23, 9, 0), qs, loose)
        self.assertIn("- [x] When did you start?", out)
        self.assertIn("    - [x] Why them? *(follow-up)*", out)
        self.assertIn("## Other questions\n\n- [ ] Anything else?", out)
        self.assertIn("- [x] What about pricing? *(asked off-list)*", out)
        self.assertNotIn("\n\n\n", out)

    def test_whisper_non_speech_labels_are_removed(self):
        self.assertEqual(_clean(" [music ends] "), "")
        self.assertEqual(_clean("Hello (laughs) there [BLANK_AUDIO]"), "Hello there")


if __name__ == "__main__":
    unittest.main()
