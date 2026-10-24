"""Regressions for the fourth review's findings (numbered 1 to 9 there)."""

import asyncio
import os
import unittest
from datetime import datetime
from pathlib import Path

import httpx

from callcoach import app, store
from callcoach.jev import MAX_BODY_BYTES, JevClient, LineCheck, request_size
from callcoach.questions import Question, parse_questions

from .test_core import SAMPLE, FakeJev, fresh_session, line


def jev_with(handler) -> JevClient:
    os.environ["CC_TEST_KEY"] = "k"
    jev = JevClient({"api_key_env": "CC_TEST_KEY"})
    jev.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return jev


class ScoringBindingTests(unittest.IsolatedAsyncioTestCase):
    """1"""

    async def asyncTearDown(self):
        app.jev = None

    async def test_a_score_for_evidence_corrected_meanwhile_is_discarded(self):
        s = fresh_session()
        s.lines = [line(0, "You", "When did you start?"), line(1, "Them", "In March.")]
        app.mark_asked("1", s.lines[0])
        gate = asyncio.Event()

        calls = {"n": 0}

        class Slow(FakeJev):
            async def score(self, lines, items, **k):
                calls["n"] += 1
                if calls["n"] == 1:  # the first answer is about the old words
                    await gate.wait()
                    return {i.key: (0.99, None) for i in items}
                return {i.key: (0.1, None) for i in items}
        app.jev = Slow()
        task = asyncio.create_task(app.score_pass(now=0.0))
        await asyncio.sleep(0)
        s.lines[1]["text"] = "In May."  # corrected while Jev was answering
        app.invalidate(s.lines[1])
        gate.set()
        await task
        self.assertEqual(s.questions[0].status, "open")

    async def test_correcting_earlier_evidence_reopens_the_tick_it_supported(self):
        s = fresh_session()
        s.lines = [line(0, "Them", "We started charging in March."), line(1, "You", "ok"),
                   line(2, "You", "When did you start?"), line(3, "Them", "As I said.")]
        app.mark_asked("1", s.lines[2])
        q = s.questions[0]
        q.early_line = 0
        q.decide("answered", "jev", 0.9)
        s.lines[0]["text"] = "We started charging in May."
        app.invalidate(s.lines[0])
        self.assertEqual(q.status, "open")


class DelayedBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Fifth review 1: a tick is reopened when a line it rested on is corrected, even after a late review
    moved the question's stretch so that line is no longer inside it."""

    def test_correction_reaches_a_tick_whose_stretch_later_shrank(self):
        s = fresh_session()
        s.lines = [line(0, "You", "When did you start?"), line(1, "Them", "Hmm."), line(2, "You", "Who was first?"),
                   line(3, "Them", "We started in March.")]
        app.mark_asked("1", s.lines[0])
        q1 = s.questions[0]
        app.decide([q1], {"1": (0.99, None)}, {"1": "r"}, app.evidence_lines(0, None))
        self.assertEqual(q1.status, "answered")
        app.mark_asked("2", s.lines[2])  # the late review: Q1's stretch now ends at line 2
        self.assertEqual(q1.stretch_end, 2)
        s.lines[3]["text"] = "We started in May."
        app.invalidate(s.lines[3])
        self.assertEqual(q1.status, "open")


class WindowBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Fifth review 2: once the latest-talk window starts rolling, unchanged results still apply and the
    asked questions are still scored."""

    async def asyncTearDown(self):
        app.jev = None

    async def test_scoring_across_the_41_to_42_line_boundary(self):
        s = fresh_session()
        s.lines = [line(i, "Them" if i % 2 else "You", f"turn {i}") for i in range(42)]
        app.mark_asked("1", s.lines[0])
        seen = []

        class Scores(FakeJev):
            async def score(self, lines, items, **k):
                seen.append(k.get("purpose", "answers"))
                return {i.key: (0.99, 0.1) for i in items}
        app.jev = Scores()
        await app.score_pass(now=100.0)
        self.assertEqual(seen, ["answers", "stretch"])
        self.assertIn("window_rev", app.cache)
        self.assertEqual(s.questions[0].status, "answered")


class FlagAfterCorrectionTests(unittest.IsolatedAsyncioTestCase):
    """Fifth review 3: correcting a flagged line before it is delivered keeps its flag."""

    async def asyncTearDown(self):
        app.jev = None

    async def test_the_lines_own_earlier_flag_does_not_suppress_it(self):
        s = fresh_session()
        app.jev = FakeJev(answer=LineCheck(worth=0.95, injection=0.0))
        s.lines = [line(0, "You", "How did you grow?", is_question=True),
                   line(1, "Them", "Akmee grew 4x.", reviewed=False, screened=False)]
        await app._review_later(s.lines[1], True)
        self.assertTrue(s.lines[1]["worth_follow_up"])
        s.state = "listening"
        await app.correct_name("Akmee", "Acme")
        await asyncio.gather(*[t for t in app.review_tasks.values() if not t.done()])
        update = await app.wait_for_update(cursor="", timeout_seconds=1)
        flagged = [l for l in update["new_lines"] if l.get("worth_follow_up")]
        self.assertEqual([l["text"] for l in flagged], ["Acme grew 4x."])


class TwoPassEvidenceTests(unittest.IsolatedAsyncioTestCase):
    """Sixth review 1: every counted pass keeps its evidence, and invalidation clears the passes."""

    def setUp(self):
        self.s = fresh_session()
        self.s.lines = [line(0, "You", "When did you start?"), line(1, "Them", "Hmm."), line(2, "You", "Who was first?"),
                        line(3, "Them", "We started in March.")]
        app.mark_asked("1", self.s.lines[0])
        self.q1 = self.s.questions[0]

    def test_reopened_tick_does_not_come_back_on_a_low_score(self):
        ev = app.evidence_lines(0, None)
        app.decide([self.q1], {"1": (0.5, None)}, {"1": "a"}, ev)
        app.decide([self.q1], {"1": (0.5, None)}, {"1": "b"}, ev)
        self.assertEqual(self.q1.status, "answered")
        app.mark_asked("2", self.s.lines[2])
        self.s.lines[3]["text"] = "We started in May."
        app.invalidate(self.s.lines[3])
        self.assertEqual((self.q1.status, self.q1.high_passes), ("open", 0))
        app.decide([self.q1], {"1": (0.1, None)}, {"1": "c"}, app.evidence_lines(0, 2))
        self.assertEqual(self.q1.status, "open")

    def test_a_pass_whose_evidence_was_later_cut_off_still_counts_as_support(self):
        app.decide([self.q1], {"1": (0.5, None)}, {"1": "a"}, app.evidence_lines(0, None))  # uses line 3
        app.mark_asked("2", self.s.lines[2])  # the late review: line 3 leaves Q1's stretch
        app.decide([self.q1], {"1": (0.5, None)}, {"1": "b"}, app.evidence_lines(0, 2))
        self.assertEqual(self.q1.status, "answered")
        self.assertIn(3, self.q1.decision_lines)
        self.s.lines[3]["text"] = "We started in May."
        app.invalidate(self.s.lines[3])
        self.assertEqual(self.q1.status, "open")


class AutoResetEvidenceTests(unittest.TestCase):
    """Seventh review P3: after Auto, a fresh tick is not reopened by evidence the old one used."""

    def test_fresh_tick_survives_correction_of_old_evidence(self):
        s = fresh_session()
        s.lines = [line(0, "You", "When did you start?"), line(1, "Them", "In March."), line(2, "You", "Who was first?"),
                   line(3, "Them", "Hmm.")]
        app.mark_asked("1", s.lines[0])
        q1 = s.questions[0]
        app.decide([q1], {"1": (0.7, None)}, {"1": "a"}, app.evidence_lines(0, None))
        app.mark_asked("2", s.lines[2])
        app.set_state("1", "auto", "you")
        app.decide([q1], {"1": (0.7, None)}, {"1": "b"}, app.evidence_lines(0, 2))
        s.lines[3]["text"] = "Hmm, yes."
        app.invalidate(s.lines[3])
        self.assertEqual(q1.status, "answered")


class SameEvidenceTests(unittest.TestCase):
    """Sixth review 2: the same evidence scored by the window and by a stretch counts once."""

    def test_one_pass_for_identical_evidence_across_purposes(self):
        s = fresh_session()
        s.lines = [line(i, "Them", "x" * 50) for i in range(5)]
        q = s.questions[1]
        q.asked = True
        rev = app.revision(s.lines)
        app.decide([q], {"2": (0.5, None)}, {"2": rev}, s.lines)  # as the window pass would
        app.decide([q], {"2": (0.5, None)}, {"2": rev}, s.lines)  # as the stretch pass would
        self.assertEqual((q.high_passes, q.status), (1, "open"))


class CooldownAfterCorrectionTests(unittest.IsolatedAsyncioTestCase):
    """Sixth review P3: re-checking an older flag does not rewind the cooldown."""

    async def asyncTearDown(self):
        app.jev = None

    async def test_a_new_flag_right_after_a_valid_one_is_still_suppressed(self):
        s = fresh_session()
        app.jev = FakeJev(answer=LineCheck(worth=0.95, injection=0.0))
        s.lines = [line(0, "You", "q1"), line(1, "Them", "Akmee grew 4x.", end=8, worth_follow_up=True,
                                                       follow_up_score=0.95),
                   line(2, "You", "q2"), line(3, "Them", "We sold for $40M.", end=38, worth_follow_up=True,
                                                    follow_up_score=0.95)]
        await app.review_line(s.lines[1])  # Q1's answer re-checked after a correction: keeps its flag
        self.assertTrue(s.lines[1]["worth_follow_up"])
        s.lines.append(line(4, "Them", "And churn halved.", end=43))
        await app.review_line(s.lines[4])
        self.assertNotIn("worth_follow_up", s.lines[4])


class MixedFailureTests(unittest.IsolatedAsyncioTestCase):
    """2"""

    async def asyncTearDown(self):
        app.jev = None

    async def test_one_failed_batch_means_the_window_is_scored_again(self):
        s = fresh_session()
        s.lines = [line(0, "Them", "We started in March.")]
        calls = {"n": 0}

        class Half(FakeJev):
            async def score(self, lines, items, **k):
                calls["n"] += 1
                return {i.key: ((0.1, 0.1) if n % 2 else (None, None)) for n, i in enumerate(items)}
        app.jev = Half()
        await app.score_pass(now=100.0)
        await app.score_pass(now=113.0)
        self.assertEqual(calls["n"], 2)

    async def test_malformed_choices_fail_cleanly(self):
        for bad in ({"choice": ["1"]}, {"choice": "1", "probabilities": {}}, {"choice": "1"},
                    {"choice": "1", "probabilities": {"1": "high"}}, {"choice": {"x": 1}, "probabilities": []}):
            def handler(request, bad=bad):
                return httpx.Response(200, json={"answers": {
                    "is_q": {"noul": 0.9}, "end": {"noul": 0.1}, "substantive": {"noul": 0.9},
                    "inject": {"noul": 0.0}, "which": bad}, "usage": {"input_tokens": 1}})
            jev = jev_with(handler)
            check = await jev.check_interviewer_turn("When?", [], parse_questions(SAMPLE).questions)
            self.assertIsNone(check, bad)
            self.assertEqual(jev.health.state, "degraded", bad)


class LongCatalogTests(unittest.TestCase):
    """3: the longest supported multibyte questions still match, in requests that fit."""

    def test_maximum_length_multibyte_catalog(self):
        jev = jev_with(lambda r: httpx.Response(200))
        qs = [Question(text=("日本語の質問です" * 73)[:582], number=n) for n in range(1, 56)]
        context = [line(i, "Them", "日本語" * 600) for i in range(6)]
        groups = jev.matching_groups("どうですか" * 300, context, qs, {})
        self.assertEqual(sum(len(g) for g in groups), 55)
        for group in groups:
            body = jev._interviewer_body("どうですか" * 300, context, group, {})
            self.assertLessEqual(request_size(body)["bytes"], MAX_BODY_BYTES)


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    """4"""

    async def test_a_cancelled_request_makes_the_cost_a_minimum(self):
        started = asyncio.Event()

        async def slow(request):
            started.set()
            await asyncio.sleep(10)
            return httpx.Response(200)
        os.environ["CC_TEST_KEY"] = "k"
        jev = JevClient({"api_key_env": "CC_TEST_KEY"})
        jev.client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
        task = asyncio.create_task(jev.check_answer_turn("q", "a"))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(jev.usage()["cost_complete"])


class SavedCorrectionTests(unittest.IsolatedAsyncioTestCase):
    """5 and 9"""

    async def asyncTearDown(self):
        app.jev = None

    async def test_a_saved_call_correction_holds_the_lifecycle_and_adds_its_cost(self):
        s = fresh_session()
        s.state, s.started = "listening", datetime.now()
        s.call_id = store.new_call_id(s.started)
        s.questions_path = Path(os.environ["CALL_COACH_HOME"]) / "q" / "r4.md"
        s.questions_path.parent.mkdir(exist_ok=True)
        s.lines = [line(0, "Them", "We built Akmee.")]
        s.usage = {"attempts": 100, "requests": 100, "failed": 0, "missing_usage": 3, "input_tokens": 1000,
                   "output_tokens": 0, "estimated_cost_usd": 0.05, "cost_complete": False, "by_purpose": {}}
        await app.save_call()
        seen = {}

        class Checking(FakeJev):
            async def check_answer_turn(self, *a, **k):
                seen["locked"] = s.lock.locked()
                return LineCheck(worth=0.1, injection=0.0)

            def usage(self):
                return {"attempts": 2, "requests": 2, "failed": 0, "missing_usage": 0, "input_tokens": 20,
                        "output_tokens": 0, "estimated_cost_usd": 0.0001, "cost_complete": True, "by_purpose": {}}

            async def aclose(self):
                pass
        real = app.new_jev
        app.new_jev = lambda: setattr(app, "jev", Checking())
        try:
            await app.correct_name("Akmee", "Acme")
        finally:
            app.new_jev = real
        self.assertTrue(seen["locked"])
        self.assertIsNone(app.jev)
        self.assertEqual((s.usage["requests"], s.usage["cost_complete"]), (102, False))


class KeyTests(unittest.TestCase):
    """6"""

    def test_a_removed_off_list_key_is_never_reused(self):
        s = fresh_session()
        s.lines = [line(0, "You", "How did you price the first enterprise contract?"),
                   line(1, "You", "What made you hire your first salesperson then?")]
        f1 = app.add_followup(s.lines[0]["text"], None, "live", asked_live=True, actor="jev")
        app.mark_asked(f1.key, s.lines[0])
        f2 = app.add_followup(s.lines[1]["text"], None, "live", asked_live=True, actor="jev")
        app.invalidate(s.lines[0])
        f3 = app.add_followup("Who signed that contract?", None, "ai")
        self.assertEqual([f.key for f in s.loose], [f2.key, f3.key])
        self.assertNotEqual(f2.key, f3.key)
        self.assertIs(s.find(f3.key), f3)


class UpdateDeliveryTests(unittest.IsolatedAsyncioTestCase):
    """7"""

    async def asyncTearDown(self):
        app.jev = None

    async def test_a_line_re_checked_after_a_correction_is_delivered_again(self):
        s = fresh_session()
        s.lines = [line(0, "Them", "Akmee grew.", review_count=1), line(1, "You", "Next?", review_count=1)]
        first = await app.wait_for_update(cursor="", timeout_seconds=1, wake_on="line")
        self.assertEqual(len(first["new_lines"]), 2)
        app.jev = FakeJev(answer=LineCheck(worth=0.95, injection=0.0))
        s.lines[0]["text"] = "Acme grew 4x."
        app.invalidate(s.lines[0])
        await app._review_later(s.lines[0], True)
        again = await app.wait_for_update(cursor=first["cursor"], timeout_seconds=1)
        self.assertEqual([l["id"] for l in again["updated_lines"]], [0])
        self.assertTrue(again["updated_lines"][0]["worth_follow_up"])

    async def test_joining_late_starts_from_the_beginning(self):
        s = fresh_session()
        s.lines = [line(0, "Them", "We grew 4x.", worth_follow_up=True)]
        cursor = (await app.get_session())["cursor"]
        update = await app.wait_for_update(cursor=cursor, timeout_seconds=1)
        self.assertEqual([l["id"] for l in update["new_lines"]], [0])


class StatusTests(unittest.TestCase):
    """8"""

    def test_jev_off_does_not_hide_a_silent_microphone(self):
        s = fresh_session()
        s.health = {"jev": {"state": "off", "message": "Jev is off; tick by hand."},
                    "capture": {"state": "degraded", "message": "No audio from the microphone."}}
        app.refresh_status()
        self.assertIn("No audio from the microphone.", s.status)
        self.assertIn("Jev is off", s.status)


if __name__ == "__main__":
    unittest.main()
