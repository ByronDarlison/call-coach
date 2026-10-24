"""Regressions for the third review's findings (numbered 1 to 12 there)."""

import asyncio
import json
import os
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx

from callcoach import app, store
from callcoach.audio import Pipeline, Segment
from callcoach.jev import (MAX_BODY_BYTES, MAX_CRITERION_UNITS, MAX_DECISIONS, MAX_STATE_UNITS, JevClient,
                           LineCheck, request_size, utf16_units)
from callcoach.questions import Question, parse_questions

from .test_core import SAMPLE, FakeJev, fresh_session, line
from .test_regressions import ctx


def jev_with(handler) -> JevClient:
    os.environ["CC_TEST_KEY"] = "k"
    jev = JevClient({"api_key_env": "CC_TEST_KEY"})
    jev.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return jev


class CoachSetupTests(unittest.IsolatedAsyncioTestCase):
    """1: a coaching token cannot choose what to load or start a recording."""

    async def test_coach_cannot_load_or_start(self):
        s = fresh_session()
        s.state = "saved"
        with self.assertRaises(RuntimeError):
            await app.load_questions_file(path=str(Path(__file__)), ctx=ctx("coach"))
        with self.assertRaises(RuntimeError):
            await app.start_call(ctx=ctx("coach"))
        self.assertEqual(s.state, "saved")


class OrderBeyondTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """2: a slow question is never placed after its answer, however long it takes."""

    async def asyncTearDown(self):
        app.pipeline = None
        for t in app.review_tasks.values():
            t.cancel()

    async def test_answer_waits_for_a_question_still_being_transcribed(self):
        s = fresh_session()
        pending = {"You": 0.0, "Them": None}
        app.pipeline = SimpleNamespace(pending_start=lambda sp: pending[sp], speaking=lambda sp: False,
                                       audio_seconds=lambda: 60.0)
        await app.on_segment(Segment("Them", "Yes.", 9.0, 9.5))
        s.held[0] = (s.held[0][0], time.monotonic() - 120)  # held far longer than any old timeout
        await app.release_held()
        self.assertEqual(s.lines, [])
        pending["You"] = None
        await app.on_segment(Segment("You", "Did you sell it before it existed?", 0.0, 8.5))
        self.assertEqual([l["speaker"] for l in s.lines], ["You", "Them"])


class InFlightStopTests(unittest.IsolatedAsyncioTestCase):
    """3: speech inside transcription when Stop times out is recorded, not lost."""

    async def test_an_in_flight_segment_is_marked_not_transcribed(self):
        got = []

        async def on_segment(seg):
            got.append(seg)

        async def on_health(*a):
            pass
        p = Pipeline(Path("/nonexistent"), Path("/nonexistent"), on_segment, on_health, whisper_port=1)

        async def slow(samples):
            await asyncio.sleep(30)
            return "never"
        p._transcribe = slow
        buf = p.buffers["M"]
        buf.samples.extend([0.1] * 16000 * 2)
        buf.has_speech, buf.speech_samples, buf.clock = True, 32000, 5.0
        buf.take(cut=False)
        p.workers = [asyncio.create_task(p._worker(b)) for b in p.buffers.values()]
        await asyncio.sleep(0.1)  # the worker is now inside transcription
        problems = await p.stop(drain_seconds=0.3)
        self.assertEqual([(g.speaker, g.failed) for g in got], [("You", True)])
        self.assertTrue(any("1 piece(s) of speech (2 s)" in x for x in problems), problems)


class CorrectionTests(unittest.IsolatedAsyncioTestCase):
    """4: corrections drop conclusions drawn from the old words; a review of old words is discarded."""

    async def asyncTearDown(self):
        app.jev = None
        for t in app.review_tasks.values():
            t.cancel()

    async def test_a_correction_reopens_jevs_tick_but_keeps_yours(self):
        s = fresh_session()
        s.lines = [line(0, "You", "When did Akmee start?", asked="1", is_question=True),
                   line(1, "Them", "In March."), line(2, "You", "Who was first?", asked="2", is_question=True),
                   line(3, "Them", "Pat.")]
        app.mark_asked("1", s.lines[0])
        app.mark_asked("2", s.lines[2])
        s.questions[0].decide("answered", "jev", 0.9)
        s.questions[1].decide("answered", "you")
        s.state = "listening"
        await app.correct_name("Akmee", "Acme")
        self.assertEqual(s.questions[0].status, "open")
        self.assertFalse(s.questions[0].asked)  # its asking is re-checked from the corrected words
        self.assertEqual(s.questions[1].status, "answered")
        self.assertNotIn("screened", s.lines[0])

    async def test_a_review_of_words_that_changed_meanwhile_is_dropped(self):
        s = fresh_session()
        gate = asyncio.Event()

        class Slow(FakeJev):
            async def check_answer_turn(self, *a, **k):
                await gate.wait()
                return LineCheck(worth=0.99, injection=0.0)
        app.jev = Slow()
        s.lines = [line(0, "Them", "Akmee grew 4x.", screened=False)]
        task = asyncio.create_task(app.review_line(s.lines[0]))
        await asyncio.sleep(0)
        s.lines[0]["text"] = "Acme grew 4x."
        gate.set()
        await task
        self.assertFalse(s.lines[0]["screened"])
        self.assertNotIn("worth_follow_up", s.lines[0])


class CacheAndValidationTests(unittest.IsolatedAsyncioTestCase):
    """5: failed scoring is retried; an empty answer is a failure, not a healthy reply."""

    async def test_an_empty_answer_is_a_failure(self):
        jev = jev_with(lambda r: httpx.Response(200, json={"answers": {"a1": {}}, "usage": {"input_tokens": 5}}))
        scores = await jev.score([line(0, "Them", "x")], parse_questions(SAMPLE).questions[:1], with_fit=False)
        self.assertEqual(scores, {"1": (None, None)})
        self.assertEqual(jev.health.state, "degraded")

    async def test_failed_latest_talk_scoring_is_tried_again(self):
        s = fresh_session()
        s.lines = [line(0, "Them", "We started in March.")]
        calls = {"n": 0}

        class Failing(FakeJev):
            async def score(self, *a, **k):
                calls["n"] += 1
                return {"1": (None, None)}
        app.jev = Failing()
        try:
            await app.score_pass(now=100.0)
            await app.score_pass(now=113.0)
        finally:
            app.jev = None
        self.assertEqual(calls["n"], 2)


class ProposalTests(unittest.IsolatedAsyncioTestCase):
    """6: a coaching proposal needs this call's id; a failed review is retried before delivery."""

    async def test_call_id_is_required_with_the_line(self):
        s = fresh_session()
        s.lines = [line(0, "Them", "We grew 4x.", worth_follow_up=True)]
        with self.assertRaises(RuntimeError):
            await app.add_question("What drove it?", 1, from_line=0, ctx=ctx("coach"))

    async def test_a_failed_review_is_retried_and_held_until_final(self):
        s = fresh_session()
        attempts = {"n": 0}

        class Flaky(FakeJev):
            async def check_answer_turn(self, *a, **k):
                attempts["n"] += 1
                return None if attempts["n"] == 1 else LineCheck(worth=0.99, injection=0.0)
        app.jev = Flaky()
        old = app.REVIEW_RETRY_AFTER
        app.REVIEW_RETRY_AFTER = 0.05
        try:
            s.lines = [line(0, "Them", "We grew 4x.", reviewed=False, screened=False)]
            task = asyncio.create_task(app._review_later(s.lines[0], True))
            await asyncio.sleep(0.01)
            self.assertEqual(app.deliverable(0), 0)  # not handed out between attempts
            await task
        finally:
            app.REVIEW_RETRY_AFTER = old
            app.jev = None
        self.assertTrue(s.lines[0]["worth_follow_up"])
        self.assertEqual(app.deliverable(0), 1)


class ByteLimitTests(unittest.TestCase):
    """7: requests are measured in the broker's own units."""

    def test_multibyte_and_emoji_requests_fit_the_broker(self):
        jev = jev_with(lambda r: httpx.Response(200))
        texts = ["日本語の質問" * 96, "🙂" * 290, ("é" * 300) + "?"]
        items = [Question(text=texts[n % 3] + f" {n}", number=n) for n in range(1, 301)]
        state = "Interview transcript. " + app.turns_json([line(i, "Them", "日本語" * 400) for i in range(40)])
        self.assertLessEqual(utf16_units(state), MAX_STATE_UNITS)
        batches = jev._batches(items, state, with_fit=True)
        self.assertEqual(sum(len(b) for b in batches), 300)
        for batch in batches:
            questions = {}
            for item in batch:
                questions.update(jev._score_questions(item, True))
            size = request_size({"model": "jev-latest", "state": state, "questions": questions})
            self.assertLessEqual(size["bytes"], MAX_BODY_BYTES)
            self.assertLessEqual(len(questions), MAX_DECISIONS)
            for q in questions.values():
                self.assertLessEqual(utf16_units(q["instructions"]), 2000)

    def test_choice_labels_fit(self):
        from callcoach.jev import fit_units
        self.assertLessEqual(utf16_units(fit_units("🙂" * 900, MAX_CRITERION_UNITS)), MAX_CRITERION_UNITS)


class CatalogTests(unittest.IsolatedAsyncioTestCase):
    """8: if part of a long list could not be searched, a turn is never called off-list."""

    async def asyncTearDown(self):
        app.jev = None

    async def test_a_failed_second_group_means_unknown(self):
        s = fresh_session()
        s.questions = [Question(text=f"Question {n}?", number=n) for n in range(1, 71)]
        calls = {"n": 0}

        class TwoGroups(FakeJev):
            async def check_interviewer_turn(self, *a, **k):
                calls["n"] += 1
                if calls["n"] == 1:
                    return LineCheck(is_question=0.99, substantive=0.99, injection=0.0, match=None)
                return None
        app.jev = TwoGroups()
        s.lines = [line(0, "You", "How did you price the first enterprise contract?", screened=False)]
        await app.review_line(s.lines[0])
        self.assertEqual(s.loose, [])


class UsageTests(unittest.IsolatedAsyncioTestCase):
    """10: an attempt that may have been billed without a receipt makes the cost a minimum."""

    async def test_timeout_then_success_is_incomplete(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("slow")
            return httpx.Response(200, json={"answers": {"worth": {"noul": 0.1}, "hook": {"choice": "none", "probabilities": {"none": 0.9}},
                                                         "inject": {"noul": 0.0}}, "usage": {"input_tokens": 10}})
        jev = jev_with(handler)
        await jev.check_answer_turn("q", "a")
        self.assertFalse(jev.usage()["cost_complete"])


class JevOffTests(unittest.TestCase):
    """12: with Jev off, the page keeps saying so all call."""

    def test_jev_off_stays_visible(self):
        s = fresh_session()
        s.base_status = "Listening."
        s.health = {"jev": {"state": "off", "message": "Jev is off; tick questions by hand."},
                    "capture": {"state": "ok", "message": "Microphone: Jabra."}}
        app.refresh_status()
        self.assertEqual((s.status, s.severity), ("Jev is off; tick questions by hand.", "warn"))


if __name__ == "__main__":
    unittest.main()
