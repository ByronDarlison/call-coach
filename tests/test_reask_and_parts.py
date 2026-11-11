"""Regressions from a real coaching call (10 Nov 2026), with names and details changed.

1. A question re-asked in other words, or put back as a summary ("so what you learned is X, is that fair?"),
   was recorded as a new off-list question, and the listed question stayed open.
2. A question with several lines, mostly answered (some of it before it was asked), stayed open.
3. "I don't know yet" ticked a question as answered.
"""

import json
import os
import unittest
from pathlib import Path

import httpx

from callcoach import app
from callcoach.jev import JevClient, LineCheck, apply_final_scores, apply_scores, part_key, thresholds
from callcoach.questions import parse_questions, render_results

from .test_core import FakeJev, line

LIST = """# Coaching call

1. Who decides who goes, you or your team leaders?
2. What did the trade show teach you about the core customer, and what has outreach produced since?
   - How many replies and sales calls
   - Were self-serve signups switched off?
3. Who should be in the room for the planning day?
"""
LIMITS = thresholds()
REASK = ("I see. Okay. So the thing that you learned from the trade show is that the CEO is on side, but the "
         "leadership team is blocking you. Is that fair to say?")


def session_with_list():
    app.session = app.Session()
    app.session.questions = parse_questions(LIST).questions
    app.session.questions_path = Path("/tmp/coaching.md")
    app.session.call_id = "20261110-090000-abcdef"
    app.session.state = "listening"
    app.limits.clear()
    app.limits.update(LIMITS)
    app.cache.clear()
    return app.session


class Reasks(FakeJev):
    """The matching check calls the turn a new question; the re-ask check gives these scores."""

    def __init__(self, scores):
        super().__init__(interviewer=LineCheck(is_question=0.97, match=None, substantive=0.97))
        self.scores = scores

    async def restates(self, text, context, items):
        return {i.key: self.scores[i.key] for i in items if i.key in self.scores}


class ReaskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.s = session_with_list()
        self.s.lines = [line(0, "You", "So what did you decide to do with self-serve?"),
                        line(1, "Them", "We left it on; it is how people find us.")]
        app.mark_asked("2", self.s.lines[0])

    async def asyncTearDown(self):
        app.jev = None

    async def test_a_summary_put_back_as_a_check_attaches_to_the_question_being_discussed(self):
        # The real scores: 0.45 to 0.47 for the question being discussed, 0.36 to 0.39 for an earlier one.
        app.jev = Reasks({"2": 0.45, "1": 0.39})
        self.s.lines.append(line(2, "You", REASK))
        await app.review_line(self.s.lines[2])
        self.assertEqual(self.s.lines[2]["asked"], "2")
        self.assertTrue(self.s.lines[2]["matched_as_reask"])
        self.assertEqual(self.s.loose, [])

    async def test_a_new_question_on_a_related_subject_still_goes_on_the_list(self):
        # "So what is their pain that you can solve?" scored 0.02 to 0.04 on the real call; the highest new
        # question in that stretch, 0.17.
        app.jev = Reasks({"2": 0.17, "1": 0.04})
        self.s.lines.append(line(2, "You", "So what is their pain that you can solve?"))
        await app.review_line(self.s.lines[2])
        self.assertEqual(len(self.s.loose), 1)
        self.assertNotIn("matched_as_reask", self.s.lines[2])

    async def test_an_earlier_question_needs_the_higher_bar(self):
        s = session_with_list()
        s.lines = [line(0, "You", "Who decides who goes?"), line(1, "Them", "Me."),
                   line(2, "You", "What did the trade show teach you?"), line(3, "Them", "Plenty.")]
        app.mark_asked("1", s.lines[0])
        app.mark_asked("2", s.lines[2])
        app.jev = Reasks({"1": 0.45})  # enough for the question being discussed, not for an earlier one
        s.lines.append(line(4, "You", "And you decided who goes yourself, is that right?"))
        await app.review_line(s.lines[4])
        self.assertEqual(len(s.loose), 1)
        app.jev.scores = {"1": 0.53}
        s.lines.append(line(5, "You", "So you decide who goes, not the team leaders, yes?"))
        await app.review_line(s.lines[5])
        self.assertEqual(s.lines[5]["asked"], "1")  # a re-ask of an earlier question returns to it
        self.assertEqual(s.current_number, 1)
        self.assertEqual(len(s.loose), 1)

    async def test_part_of_a_question_not_asked_yet_counts_as_early_evidence_and_stays_on_the_card(self):
        s = session_with_list()
        s.lines = [line(0, "You", "So how many of your core customers have you run it by?")]
        app.jev = Reasks({"2": 0.53})
        await app.review_line(s.lines[0])
        q2 = s.questions[1]
        self.assertFalse(q2.asked)
        self.assertEqual(q2.early_line, 0)
        self.assertEqual(s.lines[0]["early_for"], "2")
        self.assertEqual(s.loose, [])
        self.assertIs(s.next_item()[0], s.questions[0])  # list order is unchanged; question 2 is still to come
        app.mark_asked("2", line(5, "You", "What did the trade show teach you?"))
        self.assertEqual(app.stretch_start(q2), 0)  # once asked, the earlier talk is part of its evidence
        s.lines[0]["text"] = "So how are the kids?"  # a correction forgets the early evidence it gave
        app.invalidate(s.lines[0])
        self.assertIsNone(q2.early_line)


class RestatesRequestTests(unittest.IsolatedAsyncioTestCase):
    def jev(self, answers, sent):
        def handler(request):
            sent.update(json.loads(request.content))
            return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 10}})
        os.environ["CC_TEST_KEY"] = "k"
        jev = JevClient({"api_key_env": "CC_TEST_KEY"})
        jev.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return jev

    async def test_one_yes_or_no_per_question(self):
        sent = {}
        jev = self.jev({"r1": {"noul": 0.2}, "r2": {"noul": 0.55}, "r3": {"noul": 0.1}}, sent)
        scores = await jev.restates(REASK, [], parse_questions(LIST).questions)
        self.assertEqual(scores, {"1": 0.2, "2": 0.55, "3": 0.1})
        self.assertEqual(set(sent["questions"]), {"r1", "r2", "r3"})
        self.assertIn("is that fair", sent["questions"]["r2"]["instructions"])
        self.assertIn(REASK, sent["state"])

    async def test_a_bad_reply_matches_nothing(self):
        jev = self.jev({"r1": {"noul": 0.2}, "r2": {"noul": 7}, "r3": {"noul": 0.1}}, {})
        self.assertEqual(await jev.restates(REASK, [], parse_questions(LIST).questions), {})


class PartTests(unittest.TestCase):
    def setUp(self):
        self.q = parse_questions(LIST).questions[1]
        self.q.asked = True

    def test_lines_are_scored_only_for_a_question_with_several(self):
        one, several = parse_questions(LIST).questions[0], self.q
        self.assertEqual(list(JevClient._score_questions(one, False)), ["a1"])
        self.assertEqual(list(JevClient._score_questions(several, False)), ["a2", "a2p0", "a2p1", "a2p2"])

    def test_two_of_three_lines_answered_twice_ticks_during_the_call(self):
        scores = {"2": (0.12, None), part_key("2", 0): (0.5, None), part_key("2", 1): (0.58, None),
                  part_key("2", 2): (0.1, None)}
        apply_scores([self.q], scores, {"2": "a"}, LIMITS)
        self.assertEqual(self.q.status, "open")  # one pass is not enough...
        self.assertTrue(self.q.partial)
        apply_scores([self.q], scores, {"2": "a"}, LIMITS)
        self.assertEqual(self.q.status, "open")  # ...nor the same evidence again
        apply_scores([self.q], scores, {"2": "b"}, LIMITS)
        self.assertTrue(self.q.answered)
        self.assertEqual(self.q.parts_answered, 2)

    def test_one_line_well_over_the_bar_counts_at_once(self):
        scores = {"2": (0.1, None), part_key("2", 0): (0.7, None), part_key("2", 1): (0.65, None)}
        apply_scores([self.q], scores, {"2": "a"}, LIMITS)
        self.assertTrue(self.q.answered)

    def test_at_stop_lines_from_anywhere_in_the_call_add_up(self):
        apply_final_scores([self.q], {"2": (0.12, None), part_key("2", 0): (0.5, None),
                                      part_key("2", 2): (0.58, None)}, LIMITS)
        self.assertEqual(self.q.decided_by, "jev at stop")

    def test_one_line_of_three_is_partly_answered(self):
        apply_final_scores([self.q], {"2": (0.12, None), part_key("2", 1): (0.6, None)}, LIMITS)
        self.assertEqual(self.q.status, "open")
        self.assertTrue(self.q.partial)
        from datetime import datetime
        self.assertIn("partly answered: 1 of 3 lines", render_results("t", datetime.now(), [self.q], []))

    def test_unasked_lines_need_near_certain_evidence(self):
        q = parse_questions(LIST).questions[1]
        apply_final_scores([q], {"2": (0.1, None), part_key("2", 0): (0.6, None), part_key("2", 1): (0.6, None)},
                           LIMITS)
        self.assertEqual(q.status, "open")


class DeferralTests(unittest.TestCase):
    def test_i_dont_know_yet_is_not_an_answer(self):
        q = parse_questions("- So what is their pain that you can solve?").questions[0]
        instructions = JevClient._score_questions(q, False)["a1"]["instructions"]
        self.assertIn("'I don't know yet'", instructions)
        self.assertIn("do not: the question is still open", instructions)
        self.assertNotIn("'I don't know' with a reason counts", instructions)


class FinalEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        app.jev = None

    async def test_at_stop_an_asked_question_is_also_read_against_the_whole_call(self):
        s = session_with_list()
        s.lines = [line(0, "Them", "We talked to about fifteen of them; no contracts yet."),
                   line(1, "You", "Okay."),
                   line(2, "You", "What did the trade show teach you?"),
                   line(3, "Them", "That the CEO is on side.")]
        app.mark_asked("2", s.lines[2])
        seen = []

        class Recording(FakeJev):
            async def score(self, lines, items, **k):
                seen.append(([l["id"] for l in lines], [i.key for i in items]))
                return {}
        app.jev = Recording()
        await app.final_scores()
        self.assertTrue(any(0 in ids and "2" in keys for ids, keys in seen))


if __name__ == "__main__":
    unittest.main()
