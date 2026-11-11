"""Scripted scoring scenarios that run unchanged against any version of Call Coach's scoring.

Each scenario drives app.score_pass() with a scripted Jev: answers per scoring name, failed requests, and
things that happen while a request is out (more speech, a question asked or re-asked, Auto). It returns,
for each question key, the pass on which it was first ticked (None if never). scripts/scoring_differential.py
runs every scenario against two code trees and compares; tests/test_scoring_scenarios.py checks this
tree's results.

Only APIs present since PR 12 are used, so an older tree can be compared too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

LIST = """# Coaching call

1. Who decides who goes, you or your team leaders?
2. What did the trade show teach you about the core customer, and what has outreach produced since?
   - How many replies and sales calls
   - Were self-serve signups switched off?
3. Who should be in the room for the planning day?
"""


def line(id_, speaker, text):
    return {"id": id_, "speaker": speaker, "text": text, "raw": text, "clock": "09:00:00", "start": id_ * 5.0,
            "end": id_ * 5.0 + 3, "cut": False, "segments": 1, "reviewed": True, "screened": True}


class Scripted:
    """A stand-in for Jev's scoring. answer(name, purpose) gives a probability, or None for a failed
    request; during(purpose, names) runs while the request is out; gate, when set, holds it."""

    destination = "jev.test"

    def __init__(self, app, answer, during=None):
        from callcoach.jev import Health
        self.app, self.answer, self.during, self.gate, self.sent = app, answer, during, None, []
        self.health = Health()

    async def score(self, lines, items, with_fit=True, purpose="answers", only=None):
        from callcoach.jev import score_names, scores_from_names
        names = sorted({n for i in items for n in score_names(i, with_fit)} & (only or {
            n for i in items for n in score_names(i, with_fit)}))
        self.sent.append((purpose, names))
        if self.during:
            self.during(purpose, names)
        if self.gate:
            await self.gate.wait()
        await asyncio.sleep(0)
        out = {}
        for n in names:
            value = self.answer(n, purpose)
            if value is not None:
                out[n] = value
        return scores_from_names(items, out, with_fit)

    def usage(self):
        return {"requests": 0, "input_tokens": 0, "estimated_cost_usd": 0.0}


def setup(app, passes=None):
    from callcoach.jev import thresholds
    from callcoach.questions import parse_questions
    app.session = s = app.Session()
    s.questions = parse_questions(LIST).questions
    s.questions_path, s.call_id, s.state = Path("/tmp/scenario.md"), "20261111-130000-abcdef", "listening"
    app.limits.clear()
    app.limits.update(thresholds())
    if passes:
        app.limits["passes"] = passes
    getattr(app, "clear_scores", app.cache.clear)()
    return s


class Run:
    def __init__(self, app, s):
        self.app, self.s, self.n, self.ticks = app, s, 0, {}

    async def score(self, now):
        self.n += 1
        await self.app.score_pass(now=now)
        for q in self.s.questions:
            if q.status == "answered" and q.key not in self.ticks:
                self.ticks[q.key] = self.n


def more(s, speaker, text):
    s.lines.append(line(len(s.lines), speaker, text))
    return s.lines[-1]


# ---------- the three cases from the PR 13 review, round 2 ----------

async def continued_speech_window(app):
    """Speech keeps arriving while each window request is out; the lines sent are unchanged."""
    s = setup(app)
    s.lines = [line(0, "Them", "I decide who goes, not the team leaders."), line(1, "You", "Okay.")]
    app.jev = Scripted(app, lambda n, p: 0.99 if n == "a1" else 0.1,
                       during=lambda p, names: more(s, "Them", "And another thing.") if p == "answers" else None)
    r = Run(app, s)
    for now in (100.0, 112.0, 124.0):
        await r.score(now)
    return r.ticks


async def fit_failed_asked_during_window(app):
    """The window's fit request fails, and the question is asked while the window request is out."""
    s = setup(app)
    s.lines = [line(0, "Them", "We talked to fifteen; self-serve stays on."), line(1, "You", "Okay."),
               line(2, "Them", "And the CEO is on side."), line(3, "You", "Good."),
               line(4, "You", "What did the trade show teach you?"), line(5, "Them", "As I said.")]
    # The answer is in the earlier talk: the window scores 0.7, a stretch without it 0.1.
    app.jev = Scripted(app, lambda n, p: None if n.startswith("f") else
                       0.7 if n.startswith("a2") and p == "answers" else 0.1,
                       during=lambda p, names: app.mark_asked("2", s.lines[4]) if p == "answers" else None)
    r = Run(app, s)
    for now in (100.0, 104.0, 116.0):
        await r.score(now)
    return r.ticks


async def stretch_interleave_reask(app):
    """Question 1's stretch answers first; while question 2's request is out, question 1 is re-asked."""
    s = setup(app)
    s.lines = [line(0, "You", "Who decides who goes?"), line(1, "Them", "Me, not the team leaders."),
               line(2, "You", "What did the trade show teach you?"), line(3, "Them", "Not much.")]
    app.mark_asked("1", s.lines[0])
    app.mark_asked("2", s.lines[2])
    first = {"a1": True}

    def answer(n, p):
        if n == "a1":
            value = 0.7 if first["a1"] else 0.1
            first["a1"] = False
            return value
        return 0.1

    def during(p, names):
        if "a2" in names and len(s.lines) == 4:
            app.mark_asked("1", more(s, "You", "So you decide, not the leaders?"))
            more(s, "Them", "Yes.")
    app.jev = Scripted(app, answer, during=during)
    r = Run(app, s)
    for now in (0.0, 13.0, 26.0):
        await r.score(now)
    return r.ticks


# ---------- more parity cases: failures and things happening during waits ----------

async def normal_two_passes(app):
    s = setup(app)
    s.lines = [line(0, "You", "Who decides who goes?"), line(1, "Them", "Me.")]
    app.mark_asked("1", s.lines[0])
    app.jev = Scripted(app, lambda n, p: 0.5)
    r = Run(app, s)
    await r.score(0.0)
    more(s, "Them", "Definitely me.")
    await r.score(4.0)
    return r.ticks


async def failed_line_then_new_evidence(app):
    s = setup(app)
    s.lines = [line(0, "You", "What did the trade show teach you?"), line(1, "Them", "Plenty.")]
    app.mark_asked("2", s.lines[0])
    failed = {"a2p1": True}

    def answer(n, p):
        if n == "a2p1" and failed["a2p1"]:
            failed["a2p1"] = False
            return None
        return 0.5 if n.startswith("a2p") else 0.1
    app.jev = Scripted(app, answer)
    r = Run(app, s)
    await r.score(0.0)
    await r.score(13.0)
    more(s, "Them", "Fifteen replies, two calls, self-serve on.")
    await r.score(17.0)
    return r.ticks


async def failed_fit_window(app):
    s = setup(app)
    s.lines = [line(0, "Them", "I decide who goes."), line(1, "You", "Okay.")]
    app.jev = Scripted(app, lambda n, p: None if n == "f1" else 0.99 if n == "a1" else 0.1)
    r = Run(app, s)
    await r.score(100.0)
    more(s, "Them", "Me alone.")
    await r.score(112.0)
    await r.score(124.0)
    return r.ticks


async def reask_during_stretch_wait(app):
    s = setup(app)
    s.lines = [line(0, "You", "Who decides who goes?"), line(1, "Them", "Me.")]
    app.mark_asked("1", s.lines[0])

    def during(p, names):
        if p == "stretch" and len(s.lines) == 2:
            app.mark_asked("1", more(s, "You", "You alone?"))
            more(s, "Them", "Yes, me alone.")
    app.jev = Scripted(app, lambda n, p: 0.5, during=during)
    r = Run(app, s)
    for now in (0.0, 4.0):
        await r.score(now)
    more(s, "Them", "Nobody else decides.")
    await r.score(17.0)
    return r.ticks


async def answer_before_question(app):
    s = setup(app)
    s.lines = [line(0, "Them", "We talked to fifteen; self-serve stays on."), line(1, "You", "Okay.")]
    app.jev = Scripted(app, lambda n, p: 0.7 if n.startswith("a2") else 0.1)
    r = Run(app, s)
    await r.score(100.0)
    app.mark_asked("2", more(s, "You", "What did the trade show teach you?"))
    more(s, "Them", "As I said.")
    await r.score(104.0)
    await r.score(116.0)
    return r.ticks


# ---------- the round-4 bugs: here main's result is the defect ----------

async def auto_during_partial_retry(app):
    """Main ticks on line scores from before an Auto reset; the fix keeps the question open."""
    s = setup(app)
    s.lines = [line(0, "Them", "Good to see you."), line(1, "You", "You too."),
               line(2, "You", "What did the trade show teach you?"), line(3, "Them", "Plenty.")]
    app.mark_asked("2", s.lines[2])
    first = {"a2": 0.1, "a2p0": 0.7, "a2p1": None, "a2p2": 0.1}
    jev = app.jev = Scripted(app, lambda n, p: first.get(n, 0.1))
    r = Run(app, s)
    await r.score(0.0)
    jev.answer = lambda n, p: 0.7
    jev.gate = asyncio.Event()
    task = asyncio.create_task(r.score(13.0))
    for _ in range(5):
        await asyncio.sleep(0)
    app.set_state("2", "open", "you")
    app.set_state("2", "auto", "you")
    jev.gate.set()
    await task
    return r.ticks


async def evidence_counted_twice(app):
    """A, B, then A again with three passes needed: main counts A twice and ticks."""
    s = setup(app, passes=3)
    s.lines = [line(0, "You", "Who decides who goes?"), line(1, "Them", "Me.")]
    app.mark_asked("1", s.lines[0])
    app.jev = Scripted(app, lambda n, p: 0.5)
    r = Run(app, s)
    await r.score(0.0)
    more(s, "You", "Who should be in the room for the planning day?")
    await r.score(4.0)
    app.mark_asked("3", s.lines[2])
    await r.score(8.0)
    return r.ticks


async def evidence_closes_back(app):
    """An answer for evidence that changed meanwhile: main loses it, the fix applies it later."""
    s = setup(app)
    s.lines = [line(0, "You", "Who decides who goes?"), line(1, "Them", "Me, not the team leaders.")]
    app.mark_asked("1", s.lines[0])

    def during(p, names):
        if p == "stretch" and len(s.lines) == 2:
            more(s, "You", "Who should be in the room for the planning day?")
    app.jev = Scripted(app, lambda n, p: 0.7 if n == "a1" else 0.1, during=during)
    r = Run(app, s)
    await r.score(0.0)
    app.mark_asked("3", s.lines[2])
    await r.score(4.0)
    return r.ticks


SCENARIOS = {
    "continued_speech_window": continued_speech_window,
    "fit_failed_asked_during_window": fit_failed_asked_during_window,
    "stretch_interleave_reask": stretch_interleave_reask,
    "normal_two_passes": normal_two_passes,
    "failed_line_then_new_evidence": failed_line_then_new_evidence,
    "failed_fit_window": failed_fit_window,
    "reask_during_stretch_wait": reask_during_stretch_wait,
    "answer_before_question": answer_before_question,
    "auto_during_partial_retry": auto_during_partial_retry,
    "evidence_counted_twice": evidence_counted_twice,
    "evidence_closes_back": evidence_closes_back,
}
# Where this tree is meant to differ from main: main's tick is the defect (it must not tick here), or this
# tree recovers an answer main loses (it may tick earlier or where main never does).
MAIN_TICKS_WRONGLY = {"auto_during_partial_retry": {"2"}, "evidence_counted_twice": {"1"}}
RECOVERS_MORE = {"evidence_closes_back"}


def run_all() -> dict:
    from callcoach import app
    out = {}
    for name, scenario in SCENARIOS.items():
        try:
            out[name] = asyncio.run(scenario(app))
        finally:
            app.jev = None
    return out
