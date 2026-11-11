#!/usr/bin/env python3
"""Compare when questions tick in two versions of Call Coach's scoring, on the same scripted scenarios.

    .venv/bin/python scripts/scoring_differential.py <other tree>   # for example a checkout of origin/main

Runs tests/scoring_scenarios.py from this tree against this tree and against the other one, each in its
own process with its own data folder, and fails if this tree ticks any question later than the other, or
leaves one open that the other ticks, except where the scenarios name the other's tick as the defect.
Ticking earlier, or ticking where the other does not, is reported and allowed only where the scenarios
say this tree recovers an answer the other loses.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tests"))
import scoring_scenarios as sc  # noqa: E402


def results(tree: Path) -> dict:
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(1, sys.argv[2]); "
            "import scoring_scenarios; print(json.dumps(scoring_scenarios.run_all()))")
    with tempfile.TemporaryDirectory() as home:
        env = dict(os.environ, CALL_COACH_HOME=home, CALL_COACH_CONFIG=str(Path(home) / "none.json"))
        out = subprocess.run([sys.executable, "-c", code, str(tree), str(HERE / "tests")], cwd=tree, env=env,
                             capture_output=True, text=True, timeout=300)
    if out.returncode:
        raise SystemExit(f"scenarios failed in {tree}:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def compare(head: dict, other: dict) -> list[str]:
    problems = []
    for name in sc.SCENARIOS:
        h, o = head[name], other[name]
        wrong = sc.MAIN_TICKS_WRONGLY.get(name, set())
        for key in sorted(set(h) | set(o)):
            hp, op = h.get(key), o.get(key)
            if key in wrong:
                if hp is not None:
                    problems.append(f"{name}: question {key} ticked at pass {hp}; the other's tick is the defect")
                continue
            if op is not None and (hp is None or hp > op):
                problems.append(f"{name}: question {key} ticked at pass {hp} here, {op} in the other tree")
            elif hp is not None and (op is None or hp < op) and name not in sc.RECOVERS_MORE:
                problems.append(f"{name}: question {key} ticked at pass {hp} here, {op} in the other tree "
                                "(earlier, but not a scenario where this tree should recover more)")
    return problems


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    other_tree = Path(sys.argv[1]).resolve()
    head, other = results(HERE), results(other_tree)
    width = max(map(len, sc.SCENARIOS))
    print(f"{'scenario':{width}}  this tree        other tree")
    for name in sc.SCENARIOS:
        print(f"{name:{width}}  {json.dumps(head[name]):15}  {json.dumps(other[name])}")
    problems = compare(head, other)
    print("\n" + ("\n".join(problems) if problems else "OK: no question ticks later or less than in the other tree, "
                                                        "except where the other's tick is the defect."))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
