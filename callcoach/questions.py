"""Read a Markdown questions file and write the end-of-call results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?(.+?)\s*$")
HEADING = re.compile(r"^\s*#{2,6}\s+(.+?)\s*$")  # a single # heading is the file's title, not a section


@dataclass
class Question:
    number: int
    text: str
    section: str | None = None
    answered: bool = False
    manual: bool = False  # a hand tick or untick that Jev must not override
    answered_score: float = 0.0
    fit_score: float = 0.0
    high_passes: int = 0
    followups: list["Followup"] = field(default_factory=list)


@dataclass
class Followup:
    text: str
    source: str  # "mcp" or "you"
    answered: bool = False


def parse_questions(markdown: str) -> list[Question]:
    questions: list[Question] = []
    section = None
    for line in markdown.splitlines():
        heading = HEADING.match(line)
        if heading:
            section = heading.group(1)
            continue
        item = LIST_ITEM.match(line)
        if item:
            questions.append(Question(number=len(questions) + 1, text=item.group(1), section=section))
    return questions


def output_paths(questions_path: Path, started: datetime) -> tuple[Path, Path]:
    stamp = started.strftime("%Y-%m-%d-%H%M")
    stem = questions_path.stem
    folder = questions_path.parent
    return folder / f"{stem}-{stamp}-transcript.md", folder / f"{stem}-{stamp}-results.md"


def render_results(title: str, started: datetime, questions: list[Question], loose: list[Followup]) -> str:
    lines = [f"# {title}: results", "", f"Call started {started.strftime('%d-%b-%Y %H:%M')}.", ""]
    section = None
    for q in questions:
        if q.section and q.section != section:
            section = q.section
            if lines[-1]:
                lines.append("")
            lines += [f"## {section}", ""]
        lines.append(f"- [{'x' if q.answered else ' '}] {q.text}")
        for f in q.followups:
            lines.append(f"    - [{'x' if f.answered else ' '}] {f.text} *(follow-up)*")
    if loose:
        lines += ["", "## Follow-ups", ""]
        for f in loose:
            lines.append(f"- [{'x' if f.answered else ' '}] {f.text}")
    return "\n".join(lines).strip() + "\n"


def render_transcript(title: str, started: datetime, lines_: list[dict]) -> str:
    out = [f"# {title}: transcript", "", f"Call started {started.strftime('%d-%b-%Y %H:%M')}.", ""]
    for line in lines_:
        out.append(f"**{line['speaker']}** ({line['clock']}): {line['text']}")
        out.append("")
    return "\n".join(out).strip() + "\n"
