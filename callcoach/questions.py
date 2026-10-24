"""Read a Markdown questions file and write the end-of-call results."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

LIST_ITEM = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?(.+?)\s*$")
HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
MAX_FILE_BYTES = 256 * 1024
MAX_QUESTIONS = 300
MAX_QUESTION_CHARS = 600


@dataclass
class Item:
    """Shared fields for questions and follow-ups.

    status: "open", "answered" or "skipped". Who decided is in decided_by ("jev", "jev at stop", "you", "ai");
    manual means a person or the AI decided, and Jev leaves the item alone until it is set back to automatic.
    """

    text: str
    status: str = "open"
    manual: bool = False
    asked: bool = False  # the interviewer was heard asking it
    asked_line: int | None = None  # id of the transcript line where it was asked
    stretch_end: int | None = None  # id of the line where the interviewer moved on to another question
    answered_score: float = 0.0
    best_score: float = 0.0
    fit_score: float = 0.0
    high_passes: int = 0
    evidence_rev: str = ""  # the evidence last counted towards high_passes, so retries are not counted twice
    scored_rev: str = ""  # the evidence last scored, so unchanged evidence is not sent to Jev again
    early_line: int | None = None  # first line of earlier talk that already looked like an answer
    decision_lines: list[int] = field(default_factory=list)  # the lines Jev's tick rested on
    pass_lines: list[list[int]] = field(default_factory=list)  # the lines behind each counted pass
    decided_by: str | None = None
    decided_at: str | None = None
    decision_score: float | None = None
    skip_reason: str | None = None

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"

    @property
    def partial(self) -> bool:
        """Asked, not ticked, but Jev saw part of an answer."""
        return self.status == "open" and self.asked and self.best_score >= 0.3

    def decide(self, status: str, by: str, score: float | None = None, reason: str | None = None) -> None:
        self.status, self.decided_by, self.decision_score = status, by, score
        self.decided_at = datetime.now().strftime("%H:%M:%S")
        self.skip_reason = reason if status == "skipped" else None
        self.manual = by in ("you", "ai")

    def record(self) -> dict:
        out = asdict(self)
        out.pop("followups", None)
        out["key"] = self.key
        return out


@dataclass
class Question(Item):
    number: int = 0
    section: str | None = None
    followups: list["Followup"] = field(default_factory=list)

    @property
    def key(self) -> str:
        return str(self.number)


@dataclass
class Followup(Item):
    key_: str = ""  # "4.1" = first follow-up under question 4; "f1" = first question on its own
    source: str = "ai"  # "ai" (over MCP) or "live" (heard on the call)
    asked_live: bool = False  # asked during the call without being on the list
    from_line: int | None = None  # the flagged answer this follow-up digs into

    @property
    def key(self) -> str:
        return self.key_

    @property
    def parent_number(self) -> int | None:
        return int(self.key_.split(".")[0]) if "." in self.key_ else None


@dataclass
class Questions:
    title: str
    questions: list[Question]


def parse_questions(markdown: str, fallback_title: str = "Questions") -> Questions:
    """Every top-level list item is a question; ## headings group them; a single # heading is the title.
    Indented list items and indented lines under a question are added to that question's text.
    Anything inside a ``` code block is ignored."""
    questions: list[Question] = []
    title, section, in_fence = None, None, False
    for line in markdown.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = HEADING.match(line)
        if heading:
            if len(heading.group(1)) == 1 and title is None and not questions:
                title = heading.group(2)
            else:
                section = heading.group(2)
            continue
        item = LIST_ITEM.match(line)
        if item and not item.group(1):
            questions.append(Question(text=item.group(2), number=len(questions) + 1, section=section))
        elif questions and line.strip() and (item or line[:1].isspace()):
            extra = item.group(2) if item else line.strip()
            questions[-1].text += ("; " if item else " ") + extra
    for q in questions:
        if len(q.text) > MAX_QUESTION_CHARS:
            raise ValueError(f"Question {q.number} is longer than {MAX_QUESTION_CHARS} characters.")
    if not questions:
        raise ValueError("No questions found: write each question as a list item (- or 1.).")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"More than {MAX_QUESTIONS} questions.")
    return Questions(title=title or fallback_title, questions=questions)


def _mark(item: Item) -> str:
    return "x" if item.answered else " "


def _note(item: Item) -> str:
    if item.skipped:
        return " *(skipped" + (f": {item.skip_reason}" if item.skip_reason else "") + ")*"
    if item.answered and item.decided_by:
        return f" *(ticked by {item.decided_by})*"
    if item.partial:
        return " *(partly answered)*"
    if item.asked:
        return " *(asked, not answered)*"
    return ""


def render_results(title: str, started: datetime, questions: list[Question], loose: list[Followup],
                   usage: dict | None = None) -> str:
    lines = [f"# {title}: results", "", f"Call started {started.strftime('%d-%b-%Y %H:%M')}."]
    if usage:
        complete = usage.get("cost_complete", True)
        lines.append(f"Jev: {usage['requests']} requests, {usage['input_tokens']:,} input tokens, "
                     + (f"about US${usage['estimated_cost_usd']:.3f}." if complete else
                        f"at least US${usage['estimated_cost_usd']:.3f} (some replies had no usage receipt)."))
    lines.append("")
    section = None
    for q in questions:
        if q.section and q.section != section:
            section = q.section
            if lines[-1]:
                lines.append("")
            lines += [f"## {section}", ""]
        lines.append(f"- [{_mark(q)}] {q.text}{_note(q)}")
        for f in q.followups:
            lines.append(f"    - [{_mark(f)}] {f.text} *(follow-up)*{_note(f)}")
    if loose:
        lines += ["", "## Other questions", ""]
        for f in loose:
            tag = " *(asked off-list)*" if f.asked_live else ""
            lines.append(f"- [{_mark(f)}] {f.text}{tag}{_note(f)}")
    return "\n".join(lines).strip() + "\n"


def render_transcript(title: str, started: datetime, lines_: list[dict]) -> str:
    out = [f"# {title}: transcript", "", f"Call started {started.strftime('%d-%b-%Y %H:%M')}.", ""]
    for line in lines_:
        out.append(f"**{line['speaker']}** ({line['clock']}): {line['text']}")
        out.append("")
    return "\n".join(out).strip() + "\n"
