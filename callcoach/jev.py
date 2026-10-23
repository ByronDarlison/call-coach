"""Ask Jev which open questions have been answered and which fits best next.

Jev (TypeSafe AI) scores typed questions against a state in one request:
https://docs.typesafe.ai. Every open question gets two yes/no ("noul") checks:
has it been answered, and would it be a natural next question right now.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from .questions import Question

ANSWERED_THRESHOLD = 0.5  # live tests: real answers scored 0.6 to 0.97, non-answers 0.01 to 0.12
PASSES_TO_CHECK = 2
STATE_CHAR_LIMIT = 6000


class JevClient:
    """Calls TypeSafe directly with an API key, or a broker that holds the key.

    config["jev"] examples:
      {"url": "https://api.typesafe.ai/v1/systemone", "api_key_env": "TYPESAFE_API_KEY"}
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
            if not key:
                raise RuntimeError("No Jev key: set TYPESAFE_API_KEY or configure a broker")
            self.headers["Authorization"] = f"Bearer {key}"
        self.client = httpx.AsyncClient(timeout=15)

    async def score(self, transcript_lines: list[dict], open_questions: list[Question]) -> dict[int, tuple[float, float]]:
        if not open_questions or not transcript_lines:
            return {}
        recent = "\n".join(f"{line['speaker']}: {line['text']}" for line in transcript_lines)[-STATE_CHAR_LIMIT:]
        state = (
            "Live interview transcript, most recent last. 'You' is the interviewer asking questions; "
            "'Them' is the person being interviewed.\n\n" + recent
        )
        questions = {}
        for q in open_questions:
            questions[f"a{q.number}"] = {
                "type": "noul",
                "instructions": f"Them has already given a real answer to this question: \"{q.text}\"",
            }
            questions[f"f{q.number}"] = {
                "type": "noul",
                "instructions": (
                    f"Given the last few things Them said, this would be a natural next question to ask now: \"{q.text}\""
                ),
            }
        resp = await self.client.post(self.url, headers=self.headers,
                                      json={"model": self.model, "state": state, "questions": questions})
        resp.raise_for_status()
        answers = resp.json().get("answers", {})
        out = {}
        for q in open_questions:
            answered = float(answers.get(f"a{q.number}", {}).get("noul", 0.0))
            fit = float(answers.get(f"f{q.number}", {}).get("noul", 0.0))
            out[q.number] = (answered, fit)
        return out

    async def aclose(self) -> None:
        await self.client.aclose()


def apply_scores(questions: list[Question], scores: dict[int, tuple[float, float]]) -> bool:
    """Update question state from one Jev pass. Returns True when anything visible changed."""
    changed = False
    for q in questions:
        if q.number not in scores:
            continue
        answered, fit = scores[q.number]
        q.answered_score, q.fit_score = answered, fit
        if q.manual or q.answered:
            continue
        q.high_passes = q.high_passes + 1 if answered >= ANSWERED_THRESHOLD else 0
        if q.high_passes >= PASSES_TO_CHECK:
            q.answered = True
            changed = True
    return True if scores else changed
