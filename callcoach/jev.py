"""Ask Jev which open questions have been answered and which fits best next.

Jev (TypeSafe AI) scores typed questions against a state in one request:
https://docs.typesafe.ai. Every open question gets two yes/no ("noul") checks:
has it been answered, and would it be a natural next question right now.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx


ANSWERED_THRESHOLD = 0.5  # live tests: real answers scored 0.6 to 0.97, non-answers 0.01 to 0.12
UNASKED_THRESHOLD = 0.9  # a question nobody has asked yet needs near-certain evidence to be ticked
CLEAR_MATCH = 0.85  # a line counts as asking a listed question only when Jev is this sure
PASSES_TO_CHECK = 2
STATE_CHAR_LIMIT = 6000


class JevClient:
    """Calls TypeSafe directly with an API key, or a broker that holds the key.

    config["jev"] examples:
      {"url": "https://api.typesafe.ai/v1/systemone", "api_key_env": "TYPESAFE_API_KEY"}
      {"api_key_file": "~/.config/call-coach/jev-key"}  (written by scripts/install.sh)
      {"url": "https://<broker>/v1/systemone", "broker_secret_file": "~/.config/call-coach/broker-secret"}
    """

    def __init__(self, config: dict):
        self.url = config.get("url", "https://api.typesafe.ai/v1/systemone")
        self.model = config.get("model", "jev-latest")
        self.headers = {"Content-Type": "application/json"}
        if "broker_secret_file" in config:
            secret = Path(os.path.expanduser(config["broker_secret_file"])).read_text().strip()
            self.headers["x-broker-secret"] = secret
        else:
            key = os.environ.get(config.get("api_key_env", "TYPESAFE_API_KEY"), "")
            if not key and "api_key_file" in config:
                key_file = Path(os.path.expanduser(config["api_key_file"]))
                key = key_file.read_text().strip() if key_file.exists() else ""
            if not key:
                raise RuntimeError("No Jev key: run scripts/install.sh to save one, or set TYPESAFE_API_KEY")
            self.headers["Authorization"] = f"Bearer {key}"
        self.client = httpx.AsyncClient(timeout=15)

    async def score(self, transcript_lines: list[dict], items: list) -> dict[str, tuple[float, float]]:
        """Score open items (questions and follow-ups, each with .key and .text): answered, and fit as next."""
        if not items or not transcript_lines:
            return {}
        recent = "\n".join(f"{line['speaker']}: {line['text']}" for line in transcript_lines)[-STATE_CHAR_LIMIT:]
        state = (
            "Live interview transcript, most recent last. 'You' is the interviewer asking questions; "
            "'Them' is the person being interviewed.\n\n" + recent
        )
        questions = {}
        for item in items:
            slug = item.key.replace(".", "-")
            questions[f"a{slug}"] = {
                "type": "noul",
                "instructions": f"Them has already given a real answer to this question: \"{item.text}\"",
            }
            questions[f"f{slug}"] = {
                "type": "noul",
                "instructions": (
                    f"Given the last few things Them said, this would be a natural next question to ask now: \"{item.text}\""
                ),
            }
        resp = await self.client.post(self.url, headers=self.headers,
                                      json={"model": self.model, "state": state, "questions": questions})
        resp.raise_for_status()
        answers = resp.json().get("answers", {})
        out = {}
        for item in items:
            slug = item.key.replace(".", "-")
            out[item.key] = (float(answers.get(f"a{slug}", {}).get("noul", 0.0)),
                             float(answers.get(f"f{slug}", {}).get("noul", 0.0)))
        return out

    async def _ask(self, state: str, questions: dict) -> dict:
        resp = await self.client.post(self.url, headers=self.headers,
                                      json={"model": self.model, "state": state, "questions": questions})
        resp.raise_for_status()
        return resp.json().get("answers", {})

    async def which_question(self, line: str, items: list) -> tuple[float, str | None, float]:
        """Is the interviewer's line a question, which listed item is it (None = not on the list), and how
        sure is it that the interviewer is wrapping up the call?"""
        criteria = {item.key.replace(".", "-"): item.text for item in items}
        criteria["none"] = "None of these: a different question, or not a question"
        answers = await self._ask(f"The interviewer just said: \"{line}\"", {
            "is_q": {"type": "noul", "instructions": "The interviewer is asking the other person a question"},
            "end": {"type": "noul", "instructions": "The interviewer is wrapping up and ending the interview now"},
            "which": {"type": "choice", "criteria": criteria, "instructions":
                      "Which listed question is the interviewer asking? Same meaning counts, even if worded differently."},
        })
        which = answers.get("which", {})
        choice = which.get("choice", "none")
        sure = float(which.get("probabilities", {}).get(choice, 0.0))
        is_question = float(answers.get("is_q", {}).get("noul", 0.0))
        ending = float(answers.get("end", {}).get("noul", 0.0))
        if choice == "none":
            return is_question, None, ending
        # A listed question it is not clearly asking is neither matched nor off-list.
        return is_question, choice.replace("-", ".") if sure >= CLEAR_MATCH else "unclear", ending

    async def worth_follow_up(self, line: str) -> float:
        """How strongly the interviewee's line invites a follow-up: surprising or revealing, not routine detail."""
        answers = await self._ask(f"The interviewee just said: \"{line}\"", {
            "worth": {"type": "noul", "instructions":
                      "This answer contains something surprising, unusually revealing, or contradicting what was said "
                      "before, that a sharp interviewer would stop and dig into right now. Routine facts and figures "
                      "do not count."},
        })
        return float(answers.get("worth", {}).get("noul", 0.0))

    async def aclose(self) -> None:
        await self.client.aclose()


def apply_scores(items: list, scores: dict[str, tuple[float, float]]) -> bool:
    """Update questions and follow-ups from one Jev pass. Returns True when anything visible changed."""
    changed = False
    for item in items:
        if item.key not in scores:
            continue
        answered, fit = scores[item.key]
        item.answered_score, item.fit_score = answered, fit
        if item.manual or item.answered:
            continue
        bar = ANSWERED_THRESHOLD if getattr(item, "asked", False) or getattr(item, "asked_live", False) else UNASKED_THRESHOLD
        item.high_passes = item.high_passes + 1 if answered >= bar else 0
        if item.high_passes >= PASSES_TO_CHECK:
            item.answered = True
            changed = True
    return True if scores else changed


def apply_final_scores(items: list, scores: dict[str, tuple[float, float]]) -> None:
    """At Stop there is no second pass to wait for, so one confident score is enough to tick."""
    for item in items:
        if item.key not in scores or item.manual or item.answered:
            continue
        bar = ANSWERED_THRESHOLD if getattr(item, "asked", False) or getattr(item, "asked_live", False) else UNASKED_THRESHOLD
        if scores[item.key][0] >= bar:
            item.answered = True
