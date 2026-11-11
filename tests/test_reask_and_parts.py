"""Regressions from a real coaching call (10 Nov 2026), with names and details changed.

1. A question re-asked in other words, or put back as a summary ("so what you learned is X, is that fair?"),
   was recorded as a new off-list question, and the listed question stayed open.
2. A question with several lines, mostly answered (some of it before it was asked), stayed open.
3. "I don't know yet" ticked a question as answered.

Then from the independent review of that fix (PR 11): line ticks must keep their evidence, a hand-back to
Jev must forget line scores, one long question must not overflow a request, and partial progress must
reach the page. And from its second and third rounds (PR 12): a question's main point, lines and fit are
scored in separate requests, so any of them can fail on its own and must be retried on its own.
"""

import json
import os
import unittest
from pathlib import Path

import httpx

from callcoach import app
from callcoach.jev import (DATA_NOTICE, MAX_BODY_BYTES, MAX_DECISIONS, JevClient, LineCheck, apply_final_scores,
                           apply_scores, part_key, parts_needed, request_size, scored_completely, thresholds,
                           turns_json)
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


def long_question(number: int = 1, lines: int = 50) -> str:
    """A parser-valid question with many lines, in two-byte characters."""
    return f"{number}. " + "é" * 180 + "?\n" + "".join(f"   - {'é' * 2} {n}\n" for n in range(lines))


def split_jev(fail_if, answer=0.7):
    """A real client whose requests fail when fail_if(request questions) is true; others answer everything."""
    sent = []

    def handler(request):
        questions = json.loads(request.content)["questions"]
        sent.append(sorted(questions))
        if fail_if(questions):
            return httpx.Response(400, json={"error": "bad"})
        return httpx.Response(200, json={"answers": {k: {"noul": answer} for k in questions},
                                         "usage": {"input_tokens": 10}})
    os.environ["CC_TEST_KEY"] = "k"
    jev = JevClient({"api_key_env": "CC_TEST_KEY"}, app.limits)
    jev.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return jev, sent


class KeptScoreEvidenceTests(unittest.TestCase):
    """Round 2, 1: a line's best score under today's bar keeps its evidence, since it counts once the bar
    drops (the question is asked) or the stretch is shortened."""

    def test_unasked_scores_then_a_delayed_match_then_a_correction(self):
        s = session_with_list()
        s.lines = [line(0, "Them", "We talked to fifteen; self-serve stays on."), line(1, "You", "Okay."),
                   line(2, "Them", "More on that later."), line(3, "You", "Sure."),
                   line(4, "You", "What did the trade show teach you?"), line(5, "Them", "Not much more.")]
        q = s.questions[1]
        app.decide([q], {"2": (0.1, None), **parts(0.7, 0.7, 0.1)}, {"2": "window"}, s.lines[0:4])
        self.assertEqual(q.status, "open")  # 0.7 is under the 0.9 bar for a question not asked yet
        app.mark_asked("2", s.lines[4])  # the match lands late; the stretch starts at line 4
        app.decide([q], {"2": (0.1, None), **parts(0.1, 0.1, 0.1)}, {"2": "stretch"}, s.lines[4:6])
        self.assertTrue(q.answered)  # the kept 0.7s now clear the asked bar well
        self.assertIn(0, q.decision_lines)
        s.lines[0]["text"] = "We have not talked to anyone."  # outside the new stretch
        app.invalidate(s.lines[0])
        self.assertEqual(q.status, "open")


class IncompleteScoringTests(unittest.IsolatedAsyncioTestCase):
    """Round 2, 2: evidence counts as scored only when every piece came back; a gap is retried."""

    async def asyncTearDown(self):
        if app.jev:
            await app.jev.aclose()
        app.jev = None

    def session(self, asked: bool):
        app.session = s = app.Session()
        s.questions = parse_questions(long_question()).questions
        s.questions_path, s.call_id, s.state = Path("/tmp/long.md"), "20261111-090000-abcdef", "listening"
        app.limits.clear()
        app.limits.update(LIMITS)
        app.cache.clear()
        s.lines = [line(0, "You", "What did the trade show teach you?"), line(1, "Them", "é" * 3000),
                   line(2, "Them", "é" * 3000), line(3, "Them", "é" * 3000)]
        if asked:
            app.mark_asked("1", s.lines[0])
        return s

    def test_complete_means_every_piece(self):
        q = parse_questions(LIST).questions[1]
        full = {"2": (0.5, 0.5), **parts(0.5, 0.5, 0.5)}
        self.assertTrue(scored_completely(q, full, with_fit=True))
        self.assertFalse(scored_completely(q, {**full, part_key("2", 2): (None, None)}, with_fit=False))
        self.assertFalse(scored_completely(q, {**full, "2": (0.5, None)}, with_fit=True))  # fit missing
        self.assertTrue(scored_completely(q, {**full, "2": (0.5, None)}, with_fit=False))

    async def test_a_stretch_with_a_failed_later_batch_is_scored_again(self):
        s = self.session(asked=True)
        app.jev, sent = split_jev(lambda qs: "a1p50" in qs, answer=0.1)
        await app.score_pass(now=0.0)
        first = len(sent)
        self.assertGreater(first, 1)  # the question really was split
        self.assertEqual(s.questions[0].scored_rev, "")
        await app.score_pass(now=13.0)
        self.assertGreater(len(sent), first)  # the missing pieces of the same evidence are sent again

    async def test_a_window_with_a_failed_later_batch_is_scored_again(self):
        s = self.session(asked=False)
        app.jev, sent = split_jev(lambda qs: "a1p50" in qs or "f1" in qs, answer=0.1)
        await app.score_pass(now=100.0)
        first = len(sent)
        self.assertGreater(first, 1)
        await app.score_pass(now=200.0)
        self.assertGreater(len(sent), first)


class Recorder(FakeJev):
    """Answers every scoring question it is sent, except those `drop` names, and records what was sent."""

    def __init__(self, drop=lambda name: False, answer=0.1):
        super().__init__()
        self.drop, self.answer, self.sent = drop, answer, []

    async def score(self, lines, items, with_fit=True, purpose="answers", only=None):
        from callcoach.jev import score_names, scores_from_names
        names = {n for i in items for n in score_names(i, with_fit)}
        names = names if only is None else names & only
        self.sent.append(sorted(names))
        value = self.answer if callable(self.answer) else (lambda name: self.answer)
        return scores_from_names(items, {n: value(n) for n in names if not self.drop(n)}, with_fit)


class RetryModelTests(unittest.IsolatedAsyncioTestCase):
    """Round 3: answers are cached per evidence and scoring question, only missing ones are sent again,
    retries are throttled by the last attempt, and failing stretches take turns with the others."""

    async def asyncTearDown(self):
        if app.jev and hasattr(app.jev, "aclose"):
            await app.jev.aclose()
        app.jev = None

    def session(self, markdown: str):
        app.session = s = app.Session()
        s.questions = parse_questions(markdown).questions
        s.questions_path, s.call_id, s.state = Path("/tmp/retry.md"), "20261111-100000-abcdef", "listening"
        app.limits.clear()
        app.limits.update(LIMITS)
        app.cache.clear()
        return s

    async def test_failing_stretches_do_not_starve_a_later_question(self):
        md = "".join(f"{n}. Question {n}?\n   - first line\n   - second line\n" for n in range(1, 7)) + "7. Last one?\n"
        s = self.session(md)
        for n in range(7):
            s.lines += [line(2 * n, "You", f"Question {n + 1}?"), line(2 * n + 1, "Them", f"Answer {n + 1}.")]
            app.mark_asked(str(n + 1), s.lines[2 * n])
        # 1 to 6 stay open (a low main score) and every request about their lines fails; 7 is answered.
        app.jev = jev = Recorder(drop=lambda name: "p" in name, answer=lambda name: 0.99 if name == "a7" else 0.1)
        await app.score_pass(now=0.0)
        self.assertEqual(len(jev.sent), app.STRETCH_GROUPS_PER_PASS)  # six slots, all taken by 1 to 6
        self.assertNotIn("a7", sum(jev.sent, []))
        self.assertTrue(all(q.status == "open" for q in s.questions[:6]))
        await app.score_pass(now=4.0)  # 1 to 6 are not due again yet, so 7 gets its turn
        self.assertEqual(jev.sent[-1], ["a7"])
        self.assertTrue(s.questions[6].answered)
        await app.score_pass(now=13.0)  # then 1 to 6 retry only their missing lines
        self.assertTrue(all(all("p" in n for n in b) for b in jev.sent[7:]))

    async def test_window_line_failure_is_throttled_and_only_the_gap_is_sent(self):
        s = self.session(long_question())
        s.lines = [line(0, "Them", "é" * 3000), line(1, "Them", "é" * 3000), line(2, "Them", "é" * 3000)]
        app.jev, sent = split_jev(lambda qs: "a1p50" in qs, answer=0.1)
        await app.score_pass(now=100.0)
        first, failed = len(sent), next(b for b in sent if "a1p50" in b)
        await app.score_pass(now=104.0)
        self.assertEqual(len(sent), first)  # not again within WINDOW_EVERY, though a piece is missing
        await app.score_pass(now=112.0)
        self.assertEqual(sorted(sum(sent[first:], [])), failed)  # only the failed request's questions
        await app.score_pass(now=124.0)
        self.assertEqual(sum(sent[first:], []).count("a1"), 0)  # an answered question is never re-sent

    async def test_window_fit_failure_is_throttled_and_only_the_fit_is_sent(self):
        s = self.session(LIST)
        s.lines = [line(0, "Them", "We sold to fifteen shops."), line(1, "You", "Okay.")]
        app.jev = jev = Recorder(drop=lambda name: name == "f2")
        await app.score_pass(now=100.0)
        self.assertIn("f2", jev.sent[0])
        await app.score_pass(now=104.0)
        self.assertEqual(len(jev.sent), 1)
        jev.drop = lambda name: False
        await app.score_pass(now=112.0)
        self.assertEqual(jev.sent[1], ["f2"])
        for now in (124.0, 136.0, 148.0):
            await app.score_pass(now=now)
        self.assertEqual(len(jev.sent), 2)  # complete now: unchanged evidence is not sent again

    async def test_stretch_line_failure_is_throttled_and_only_the_gap_is_sent(self):
        s = self.session(LIST)
        s.lines = [line(0, "Them", "Good to see you."), line(1, "You", "You too."),
                   line(2, "You", "What did the trade show teach you?"), line(3, "Them", "Plenty.")]
        app.mark_asked("2", s.lines[2])  # its stretch is not the same evidence as the window
        app.jev = jev = Recorder(drop=lambda name: name == "a2p1")

        def stretch():  # the requests about question 2 (the others are the window's, for 1 and 3)
            return [b for b in jev.sent if any(n.startswith("a2") for n in b)]
        await app.score_pass(now=0.0)
        self.assertEqual(stretch(), [["a2", "a2p0", "a2p1", "a2p2"]])
        await app.score_pass(now=4.0)
        self.assertEqual(len(stretch()), 1)
        jev.drop = lambda name: False
        await app.score_pass(now=13.0)
        self.assertEqual(stretch()[1], ["a2p1"])
        self.assertEqual(s.questions[1].scored_rev, app.revision(app.evidence_lines(2, None)))
        for now in (26.0, 39.0, 52.0):
            await app.score_pass(now=now)
        self.assertEqual(len(stretch()), 2)  # complete: unchanged evidence is never sent again

    async def test_a_correction_drops_the_cached_answers(self):
        s = self.session(LIST)
        s.lines = [line(0, "You", "What did the trade show teach you?"), line(1, "Them", "Plenty.")]
        app.mark_asked("2", s.lines[0])
        app.jev = jev = Recorder()
        await app.score_pass(now=0.0)
        s.lines[1]["text"] = "Plenty, and self-serve stays on."
        app.invalidate(s.lines[1])
        await app.score_pass(now=1.0)  # new evidence: sent at once, in full
        self.assertEqual(jev.sent[-1], ["a2", "a2p0", "a2p1", "a2p2"])


class LinesWithoutMainScoreTests(unittest.IsolatedAsyncioTestCase):
    """Round 2, 3: enough answered lines tick a question even when its main score failed."""

    def setUp(self):
        self.q = parse_questions(LIST).questions[1]
        self.q.asked = True

    def test_during_the_call(self):
        self.assertTrue(apply_scores([self.q], parts(0.7, 0.7, 0.1), {"2": "a"}, LIMITS))
        self.assertTrue(self.q.answered)
        self.assertEqual(self.q.decision_score, 0.7)

    def test_at_stop(self):
        apply_final_scores([self.q], parts(0.5, 0.5, 0.1), LIMITS)
        self.assertEqual(self.q.decided_by, "jev at stop")

    def test_manual_and_ticked_items_are_still_left_alone(self):
        self.q.decide("open", "you")
        apply_scores([self.q], parts(0.9, 0.9, 0.9), {"2": "a"}, LIMITS)
        apply_final_scores([self.q], parts(0.9, 0.9, 0.9), LIMITS)
        self.assertEqual((self.q.status, self.q.decided_by), ("open", "you"))

    async def test_a_real_split_request_with_a_failed_first_batch(self):
        lines = [line(i, "Them", "é" * 400) for i in range(60)]
        state = ("Interview transcript, most recent last. 'You' is the interviewer; 'Them' is the person being "
                 "interviewed. " + DATA_NOTICE + "\n\n" + turns_json(lines))
        os.environ["CC_TEST_KEY"] = "k"
        probe = JevClient({"api_key_env": "CC_TEST_KEY"})
        # Find a list where question 2's main point is in the first request but most of its lines are not.
        for k in range(2, 50):
            qs = parse_questions(long_question(1, k) + long_question(2)).questions
            first = probe._batches(qs, state, with_fit=False)[0]
            if "a2" in first and 51 - sum(key.startswith("a2p") for key in first) >= parts_needed(51, LIMITS):
                break
        else:
            self.fail("no list puts question 2's main point in the first request with most lines after it")
        await probe.aclose()
        jev, sent = split_jev(lambda q: "a2" in q)
        scores = await jev.score(lines, qs, with_fit=False, purpose="final")
        await jev.aclose()
        self.assertGreater(len(sent), 1)
        self.assertIsNone(scores["2"][0])  # the main point's request failed
        for q in qs:
            q.asked = True
        apply_final_scores(qs, scores, LIMITS)
        self.assertEqual(qs[1].decided_by, "jev at stop")


if __name__ == "__main__":
    unittest.main()
