"""The last check at Stop (issue 14): it scores every open item one at a time, keeps what came back when it is
cut short and says so, ticks a question when one of its asks is clearly answered anywhere in the call, and
never ticks an item the questions file marks "(Tick by hand.)"."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from callcoach import app, store
from callcoach.jev import NUMBERS, JevClient, any_key, apply_final_scores, apply_scores, by_hand, part_key, thresholds
from callcoach.questions import parse_questions

from .test_core import FakeJev, fresh_session, line

LIMITS = thresholds()
LIST = """# Placeholder interview

- What does the product cost, and who pays for it?
- Read it back to them and check it. (Tick by hand.)
- What is the plan for next year?
    - Then: who owns it?
"""


def session_with(text=LIST):
    s = fresh_session()
    s.questions = parse_questions(text).questions
    return s


class ByHandTests(unittest.TestCase):
    def test_an_item_marked_tick_by_hand_is_never_ticked_by_jev(self):
        q = parse_questions(LIST).questions[1]
        self.assertTrue(by_hand(q))
        q.asked = True
        apply_scores([q], {"2": (0.99, None)}, {"2": "a"}, LIMITS)
        apply_scores([q], {"2": (0.99, None)}, {"2": "b"}, LIMITS)
        apply_final_scores([q], {"2": (0.99, None), any_key("2"): (0.99, None)}, LIMITS)
        self.assertEqual(q.status, "open")
        self.assertEqual(q.best_score, 0.99)  # still scored, so the page can show it was heard

    def test_a_line_marked_tick_by_hand_holds_the_whole_question_for_you(self):
        q = parse_questions("- Ask the first thing.\n    - Then: check it with them. (Tick by hand.)").questions[0]
        q.asked = True
        apply_final_scores([q], {"1": (0.9, None), part_key("1", 0): (0.9, None), part_key("1", 1): (0.9, None)},
                           LIMITS)
        self.assertEqual(q.status, "open")

    def test_a_person_can_still_tick_it(self):
        s = session_with()
        app.set_state("2", "answered", "you")
        self.assertEqual(s.questions[1].status, "answered")


class AnyAskTests(unittest.TestCase):
    def test_one_ask_clearly_answered_anywhere_ticks_at_stop_asked_or_not(self):
        q = parse_questions(LIST).questions[0]
        apply_final_scores([q], {"1": (0.3, None), any_key("1"): (0.6, None)}, LIMITS)
        self.assertEqual(q.status, "open")
        apply_final_scores([q], {"1": (0.3, None), any_key("1"): (LIMITS["stop_any"], None)}, LIMITS)
        self.assertEqual((q.status, q.decided_by), ("answered", "jev at stop"))
        self.assertEqual(q.decision_score, LIMITS["stop_any"])

    def test_the_stop_question_is_asked_only_at_stop(self):
        q = parse_questions(LIST).questions[0]
        live = JevClient._score_questions(q, False)
        self.assertEqual(set(live), {"a1"})
        stop = JevClient._score_questions(q, False, any_ask=True)
        self.assertIn("at least one of the things this question asks", stop["s1"]["instructions"])
        self.assertIn(NUMBERS, stop["a1"]["instructions"])  # an amount asked for needs an amount given


class StopPassTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        app.jev = None

    def call(self):
        s = session_with()
        s.lines = [line(0, "You", "Tell me about the product."), line(1, "Them", "It costs a fixed amount a month."),
                   line(2, "You", "What is the plan for next year?"), line(3, "Them", "To grow, and I own it.")]
        app.mark_asked("3", s.lines[2])
        return s

    async def test_each_item_is_scored_on_its_own_with_the_stop_question(self):
        s = self.call()
        seen = []

        class Recording(FakeJev):
            async def score(self, lines, items, **k):
                seen.append(([i.key for i in items], k.get("any_ask"), k.get("purpose")))
                return {i.key: (0.1, None) for i in items}
        app.jev = Recording()
        self.assertIsNone(await app.final_scores())
        self.assertTrue(seen)
        self.assertTrue(all(len(keys) == 1 and any_ask and purpose == "final" for keys, any_ask, purpose in seen))
        self.assertEqual({k for keys, _, _ in seen for k in keys}, {"1", "2", "3"})
        self.assertEqual(s.questions[0].status, "open")

    async def test_a_pass_cut_short_keeps_what_came_back_and_says_so(self):
        s = self.call()

        class Slow(FakeJev):
            async def score(self, lines, items, **k):
                if items[0].key == "3":
                    await asyncio.sleep(30)
                return {i.key: (0.1, None) for i in items} | {any_key(i.key): (0.9, None) for i in items}
        app.jev = Slow()
        problem = await app.final_scores(timeout=0.5)
        self.assertEqual(s.questions[0].decided_by, "jev at stop")
        self.assertEqual(s.questions[1].status, "open")  # by hand, whatever the score
        self.assertEqual(s.questions[2].status, "open")
        self.assertIn("time limit", problem)

    async def test_checks_jev_did_not_answer_are_reported(self):
        self.call()

        class Failing(FakeJev):
            async def score(self, lines, items, **k):
                return {i.key: (None, None) for i in items}
        app.jev = Failing()
        self.assertIn("no answer", await app.final_scores())

    async def test_stop_saves_the_problem_with_the_call(self):
        s = self.call()
        folder = Path(tempfile.mkdtemp(prefix="cc-stop-"))
        s.questions_path = folder / "questions.md"
        s.questions_path.write_text(LIST)
        from datetime import datetime
        s.started = datetime.now()

        class Failing(FakeJev):
            async def score(self, lines, items, **k):
                return {i.key: (None, None) for i in items}

            async def aclose(self):
                pass
        app.jev = Failing()
        await app.stop_session()
        self.assertEqual(s.state, "saved")
        saved = json.loads((store.call_dir(s.call_id) / "results.json").read_text())
        self.assertTrue(any("incomplete" in p for p in saved["problems"]))


if __name__ == "__main__":
    unittest.main()
