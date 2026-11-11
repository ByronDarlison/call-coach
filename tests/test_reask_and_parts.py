"""Regressions from a real coaching call (10 Nov 2026), with names and details changed.

1. A question re-asked in other words, or put back as a summary ("so what you learned is X, is that fair?"),
   was recorded as a new off-list question, and the listed question stayed open.
2. A question with several lines, mostly answered (some of it before it was asked), stayed open.
3. "I don't know yet" ticked a question as answered.

Then from the independent review of that fix (PR 11): line ticks must keep their evidence, a hand-back to
Jev must forget line scores, one long question must not overflow a request, and partial progress must
reach the page.
"""

import json
import os
import unittest
from pathlib import Path

import httpx

from callcoach import app
from callcoach.jev import (MAX_BODY_BYTES, MAX_DECISIONS, JevClient, LineCheck, apply_final_scores, apply_scores,
                           part_key, request_size, thresholds, turns_json)
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


def parts(*scores, key="2"):
    return {part_key(key, n): (v, None) for n, v in enumerate(scores)}


class PartEvidenceTests(unittest.IsolatedAsyncioTestCase):
    """Review 1: a tick from line scores keeps every line of evidence behind it."""

    async def asyncTearDown(self):
        app.jev = None

    async def test_a_tick_keeps_line_evidence_from_a_longer_earlier_stretch(self):
        s = session_with_list()
        s.lines = [line(0, "Them", "We talked to about fifteen; no contracts."), line(1, "You", "Okay."),
                   line(2, "You", "What did the trade show teach you?"), line(3, "Them", "The CEO is on side."),
                   line(4, "Them", "Self-serve stays on."), line(5, "You", "Right.")]
        app.mark_asked("2", s.lines[2])
        q = s.questions[1]
        app.decide([q], {"2": (0.1, None), **parts(0.5, 0.55, 0.1)}, {"2": "long"}, s.lines[0:6])
        self.assertEqual(q.status, "open")  # one pass per line so far
        # A delayed match shortened the stretch: the second pass sees only lines 2 and 3.
        app.decide([q], {"2": (0.1, None), **parts(0.5, 0.55, 0.1)}, {"2": "short"}, s.lines[2:4])
        self.assertTrue(q.answered)
        self.assertIn(0, q.decision_lines)
        s.lines[0]["text"] = "We have not talked to anyone yet."  # outside the stretch the tick now rests on
        app.invalidate(s.lines[0])
        self.assertEqual(q.status, "open")

    async def test_at_stop_an_answer_outside_the_stretch_is_tracked(self):
        s = session_with_list()
        s.lines = [line(0, "Them", "We talked to about fifteen; self-serve stays on."), line(1, "You", "Okay."),
                   line(2, "You", "What did the trade show teach you?"), line(3, "Them", "Not much."),
                   line(4, "You", "Who should be in the room?"), line(5, "Them", "Not sure.")]
        app.mark_asked("2", s.lines[2])
        app.mark_asked("3", s.lines[4])

        class Outside(FakeJev):
            async def score(self, lines, items, **k):
                ids = [l["id"] for l in lines]
                hit = 0.6 if 0 in ids else 0.1
                out = {i.key: (0.1, None) for i in items}
                out.update(parts(0.1, hit, hit))
                return out
        app.jev = Outside()
        await app.final_scores()
        q = s.questions[1]
        self.assertEqual(q.decided_by, "jev at stop")
        self.assertIn(0, q.decision_lines)
        s.lines[0]["text"] = "We have not started yet."
        app.invalidate(s.lines[0])
        self.assertEqual(q.status, "open")


class AutoTests(unittest.TestCase):
    """Review 2: handing a question back to Jev forgets its line scores."""

    def test_zero_scores_after_auto_leave_it_open(self):
        s = session_with_list()
        s.lines = [line(0, "You", "What did the trade show teach you?"), line(1, "Them", "Plenty.")]
        app.mark_asked("2", s.lines[0])
        q = s.questions[1]
        app.decide([q], {"2": (0.1, None), **parts(0.7, 0.7, 0.0)}, {"2": "a"}, s.lines)
        self.assertTrue(q.answered)
        app.set_state("2", "auto", "you")
        self.assertEqual((q.part_best, q.part_hits, q.part_revs, q.part_lines, q.parts_answered), ([], [], [], [], 0))
        app.decide([q], {"2": (0.0, None), **parts(0.0, 0.0, 0.0)}, {"2": "b"}, s.lines)
        self.assertEqual(q.status, "open")


class LongQuestionTests(unittest.IsolatedAsyncioTestCase):
    """Review 3: one question written on many lines must not overflow the broker's request limits."""

    def long_question(self):
        # Parser-valid: 50 lines, under the 600-character limit, in two-byte characters.
        text = "1. " + "é" * 180 + "?\n" + "".join(f"   - {'é' * 2} {n}\n" for n in range(50))
        q = parse_questions(text).questions[0]
        self.assertEqual(len(q.parts), 51)
        return q

    def full_state(self):
        lines = [line(i, "Them", "é" * 400) for i in range(60)]  # more than the evidence limit holds
        return ("Interview transcript, most recent last. 'You' is the interviewer; 'Them' is the person being "
                "interviewed. " + __import__("callcoach.jev", fromlist=["DATA_NOTICE"]).DATA_NOTICE + "\n\n"
                + turns_json(lines)), lines

    def jev(self, handler):
        os.environ["CC_TEST_KEY"] = "k"
        jev = JevClient({"api_key_env": "CC_TEST_KEY"})
        jev.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return jev

    def test_every_request_fits_even_for_one_question(self):
        q = self.long_question()
        state, _ = self.full_state()
        batches = self.jev(lambda r: httpx.Response(200))._batches([q], state, with_fit=True)
        self.assertGreater(len(batches), 1)
        self.assertEqual(sum(len(b) for b in batches), 53)  # the answer, its fit and 51 lines
        for body in batches:
            self.assertLessEqual(request_size({"model": "jev-latest", "state": state, "questions": body})["bytes"],
                                 MAX_BODY_BYTES)
            self.assertLessEqual(len(body), MAX_DECISIONS)

    def test_many_long_questions_stay_under_the_decision_limit(self):
        qs = [parse_questions("1. Q?\n" + "".join(f"   - line {n}\n" for n in range(40))).questions[0]
              for _ in range(6)]
        for n, q in enumerate(qs):
            q.number = n + 1
        batches = self.jev(lambda r: httpx.Response(200))._batches(qs, "state", with_fit=True)
        self.assertEqual(sum(len(b) for b in batches), 6 * 43)
        self.assertTrue(all(len(b) <= MAX_DECISIONS for b in batches))
        self.assertGreater(len(batches), 1)

    async def test_answers_from_several_requests_are_merged(self):
        sizes = []

        def handler(request):
            body = json.loads(request.content)
            sizes.append(len(request.content))
            return httpx.Response(200, json={"answers": {k: {"noul": 0.5} for k in body["questions"]},
                                             "usage": {"input_tokens": 10}})
        q = self.long_question()
        _, lines = self.full_state()
        scores = await self.jev(handler).score(lines, [q], with_fit=False)
        self.assertGreater(len(sizes), 1)
        self.assertTrue(all(n <= MAX_BODY_BYTES for n in sizes))
        self.assertEqual(scores["1"], (0.5, None))
        self.assertTrue(all(scores[part_key("1", n)] == (0.5, None) for n in range(51)))


class PartialUpdateTests(unittest.TestCase):
    """Review 4: partial progress is a visible change, so the page is told."""

    def test_a_line_answered_without_a_tick_counts_as_a_change(self):
        q = parse_questions(LIST).questions[1]
        q.asked = True
        self.assertTrue(apply_scores([q], {"2": (0.1, None), **parts(0.7, 0.1, 0.1)}, {"2": "a"}, LIMITS))
        self.assertEqual((q.status, q.parts_answered), ("open", 1))
        self.assertFalse(apply_scores([q], {"2": (0.1, None), **parts(0.7, 0.1, 0.1)}, {"2": "b"}, LIMITS))


if __name__ == "__main__":
    unittest.main()
