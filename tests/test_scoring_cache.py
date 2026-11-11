"""Regressions from the fourth review of the line-scoring work (after PR 12).

1. A retry still on its way when the cache is cleared (a hand-back to Jev, a correction) must not bring
   back scores from before the clear.
2. Cached answers are applied whenever the evidence they belong to is current, separately from deciding
   what to send: answers that came back for evidence that changed meanwhile, and answers cached before a
   question was asked, must still be able to tick it.

And from the review of that fix (PR 13): a question asked while the window request is out keeps the
window's early evidence, and evidence a stretch closes back to is not counted twice.
"""

import asyncio
import unittest
from pathlib import Path

from callcoach import app
from callcoach.jev import score_names, scores_from_names, thresholds
from callcoach.questions import parse_questions

from .test_core import FakeJev, line

LIST = """# Coaching call

1. Who decides who goes, you or your team leaders?
2. What did the trade show teach you about the core customer, and what has outreach produced since?
   - How many replies and sales calls
   - Were self-serve signups switched off?
3. Who should be in the room for the planning day?
"""


def session():
    app.session = s = app.Session()
    s.questions = parse_questions(LIST).questions
    s.questions_path, s.call_id, s.state = Path("/tmp/cache.md"), "20261111-120000-abcdef", "listening"
    app.limits.clear()
    app.limits.update(thresholds())
    app.clear_scores()
    return s


class Scripted(FakeJev):
    """Answers each scoring name with answer(name) (None: that request failed), records what was sent, and
    can wait on a gate or run a hook while the request is on its way."""

    def __init__(self, answer):
        super().__init__()
        self.answer, self.sent, self.gate, self.during = answer, [], None, None

    async def score(self, lines, items, with_fit=True, purpose="answers", only=None):
        names = {n for i in items for n in score_names(i, with_fit)}
        names = sorted(names if only is None else names & only)
        self.sent.append((purpose, names))
        if self.during:
            self.during(purpose)
        if self.gate:
            await self.gate.wait()
        answers = {n: v for n in names if (v := self.answer(n)) is not None}
        return scores_from_names(items, answers, with_fit)


class InFlightClearTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        app.jev = None

    async def test_auto_during_a_partial_retry_does_not_bring_back_old_line_scores(self):
        s = session()
        s.lines = [line(0, "Them", "Good to see you."), line(1, "You", "You too."),
                   line(2, "You", "What did the trade show teach you?"), line(3, "Them", "Plenty.")]
        app.mark_asked("2", s.lines[2])
        q = s.questions[1]
        first = {"a2": 0.1, "a2p0": 0.7, "a2p1": None, "a2p2": 0.1}  # the request for line 2 fails
        app.jev = jev = Scripted(lambda n: first.get(n, 0.1))
        await app.score_pass(now=0.0)
        self.assertEqual(q.status, "open")  # one line of three answered
        # The retry asks only for line 2; while it is on its way, Open then Auto clears every line score.
        jev.answer = lambda n: 0.7
        jev.gate = asyncio.Event()
        task = asyncio.create_task(app.score_pass(now=13.0))
        await asyncio.sleep(0)
        self.assertIn(("stretch", ["a2p1"]), jev.sent)
        app.set_state("2", "open", "you")
        app.set_state("2", "auto", "you")
        jev.gate.set()
        await task
        self.assertEqual(q.status, "open")  # 0.7 on line 2 alone, or with the forgotten line 1, ticks nothing
        rev = app.revision(app.evidence_lines(2, None))
        self.assertEqual(app.missing(rev, [q], False), set(score_names(q, False)))  # the late answer was dropped

    async def test_a_correction_during_a_window_request_drops_its_answers(self):
        s = session()
        s.lines = [line(0, "Them", "We decided who goes ourselves."), line(1, "You", "Okay.")]
        app.jev = jev = Scripted(lambda n: 0.95)
        jev.gate = asyncio.Event()
        task = asyncio.create_task(app.score_pass(now=100.0))
        await asyncio.sleep(0)
        s.lines[0]["text"] = "We have not decided."
        app.invalidate(s.lines[0])
        jev.gate.set()
        await task
        self.assertEqual(app.cache.get("answers", {}), {})
        self.assertTrue(all(q.status == "open" for q in s.questions))


class ApplyCachedTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        app.jev = None

    async def test_answers_for_evidence_that_changed_meanwhile_apply_when_it_is_current_again(self):
        s = session()
        s.lines = [line(0, "You", "Who decides who goes?"), line(1, "Them", "Me, not the team leaders.")]
        app.mark_asked("1", s.lines[0])

        def extend(purpose):
            # While question 1's answer is on its way, the interviewer's next turn lands and extends its stretch.
            if purpose == "stretch" and len(s.lines) == 2:
                s.lines.append(line(2, "You", "Who should be in the room for the planning day?"))
        app.jev = jev = Scripted(lambda n: 0.7 if n == "a1" else 0.1)
        jev.during = extend
        await app.score_pass(now=0.0)
        q1 = s.questions[0]
        self.assertEqual(q1.status, "open")  # the answer was for evidence that is no longer current
        app.mark_asked("3", s.lines[2])  # a delayed match ends question 1's stretch where it was
        await app.score_pass(now=4.0)
        self.assertTrue(q1.answered)  # the cached answer applies now
        self.assertEqual(sum(names.count("a1") for _, names in jev.sent), 1)  # and was not asked again

    async def test_answers_cached_before_a_question_was_asked_count_once_it_is(self):
        s = session()
        # Early in a call: the answer comes first, and the question is asked afterwards.
        s.lines = [line(0, "Them", "We talked to fifteen; self-serve stays on."), line(1, "You", "Okay."),
                   line(2, "Them", "And the CEO is on side."), line(3, "You", "Good."),
                   line(4, "You", "What did the trade show teach you?"), line(5, "Them", "As I said.")]
        app.jev = jev = Scripted(lambda name: 0.7 if name.startswith("a2") else 0.1)
        await app.score_pass(now=100.0)
        q2 = s.questions[1]
        self.assertEqual(q2.status, "open")  # 0.7 is under the 0.9 bar for a question not asked yet
        window_rev = app.revision(app.evidence_lines(0, None))
        app.mark_asked("2", s.lines[4])  # with the earlier talk, its stretch is exactly the evidence scored
        self.assertEqual(app.revision(app.evidence_lines(app.stretch_start(q2), q2.stretch_end)), window_rev)
        sent = len(jev.sent)
        await app.score_pass(now=104.0)
        self.assertTrue(q2.answered)  # the same answers, now against the bar for an asked question
        self.assertFalse(any(n.startswith("a2") for _, names in jev.sent[sent:] for n in names))

    async def test_applying_the_same_answers_again_adds_no_pass(self):
        s = session()
        s.lines = [line(0, "You", "Who decides who goes?"), line(1, "Them", "Me.")]
        app.mark_asked("1", s.lines[0])
        app.jev = Scripted(lambda n: 0.5)  # over the 0.45 bar, not by enough to tick in one pass
        for now in (0.0, 4.0, 8.0, 12.0, 16.0):
            await app.score_pass(now=now)
        q1 = s.questions[0]
        self.assertEqual((q1.status, q1.high_passes), ("open", 1))
        self.assertEqual(len(q1.pass_lines), 1)


class AskedDuringWindowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        app.jev = None

    async def test_a_question_asked_while_the_window_is_out_keeps_its_early_evidence(self):
        s = session()
        s.lines = [line(0, "Them", "We talked to fifteen; self-serve stays on."), line(1, "You", "Okay."),
                   line(2, "Them", "And the CEO is on side."), line(3, "You", "Good."),
                   line(4, "You", "What did the trade show teach you?"), line(5, "Them", "As I said.")]
        app.jev = jev = Scripted(lambda name: 0.7 if name.startswith(("a2", "f2")) else 0.1)
        jev.during = lambda purpose: app.mark_asked("2", s.lines[4]) if purpose == "answers" else None
        await app.score_pass(now=100.0)
        q2 = s.questions[1]
        self.assertEqual(q2.early_line, 0)  # the window's answers still mark the earlier talk
        self.assertTrue(q2.answered)  # its stretch is that window: the cached answers count at the asked bar
        self.assertEqual([p for p, _ in jev.sent], ["answers"])  # no stretch request was needed


class CountedOnceTests(unittest.IsolatedAsyncioTestCase):
    """Evidence a stretch closes back to (A, then B, then A again) counts once."""

    async def asyncTearDown(self):
        app.jev = None
        app.limits["passes"] = thresholds()["passes"]

    async def a_b_a(self, key: str, answer):
        s = session()
        app.limits["passes"] = 3
        s.lines = [line(0, "You", "Question?"), line(1, "Them", "An answer.")]
        app.mark_asked(key, s.lines[0])
        app.jev = Scripted(answer)
        await app.score_pass(now=0.0)  # A: lines 0 and 1
        s.lines.append(line(2, "You", "Who should be in the room for the planning day?"))
        await app.score_pass(now=4.0)  # B: lines 0 to 2
        app.mark_asked("3", s.lines[2])  # a delayed match closes the stretch back to A
        await app.score_pass(now=8.0)  # A again, from the cache
        return s.questions[int(key) - 1]

    async def test_the_whole_question(self):
        q = await self.a_b_a("1", lambda n: 0.5)  # over the bar, not enough for one pass alone
        self.assertEqual((q.high_passes, q.status), (2, "open"))
        self.assertEqual(len(q.pass_lines), 2)

    async def test_each_line(self):
        q = await self.a_b_a("2", lambda n: 0.5 if n in ("a2p0", "a2p1") else 0.1)
        self.assertEqual(q.part_hits[:2], [2, 2])
        self.assertEqual(q.status, "open")

    def test_auto_and_corrections_forget_what_was_counted(self):
        s = session()
        s.lines = [line(0, "You", "Question?"), line(1, "Them", "An answer.")]
        app.mark_asked("2", s.lines[0])
        q = s.questions[1]
        app.decide([q], {"2": (0.5, None), "2/p0": (0.5, None)}, {"2": "A"}, s.lines)
        self.assertEqual((q.counted_revs, q.part_counted[0]), (["A"], ["A"]))
        app.set_state("2", "auto", "you")
        self.assertEqual((q.counted_revs, q.part_counted), ([], []))
        app.decide([q], {"2": (0.5, None), "2/p0": (0.5, None)}, {"2": "A"}, s.lines)
        s.lines[1]["text"] = "A different answer."
        app.invalidate(s.lines[1])
        self.assertEqual((q.counted_revs, q.part_counted), ([], []))


if __name__ == "__main__":
    unittest.main()
