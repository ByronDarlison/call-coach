"""The real service, run as a separate process with the stand-in helpers in tests/fakes:
closing the launcher window (hangup) or a TERM signal during a call saves the call before exiting, and a
coaching token is refused owner tools through the real MCP route."""

import json
import os
import pty
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAKES = ROOT / "tests" / "fakes"


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cc-shutdown-"))
        (self.tmp / "home" / "models").mkdir(parents=True)
        (self.tmp / "home" / "models" / "ggml-small.en.bin").write_bytes(b"fake")
        (self.tmp / "questions.md").write_text("- When did you start?\n- Who was first?\n")
        (self.tmp / "config.json").write_text(json.dumps({"capture_bin": str(FAKES / "cc-capture"),
                                                          "jev": {"api_key_env": "CC_NONE"}}))
        self.env = dict(os.environ, CALL_COACH_HOME=str(self.tmp / "home"), CALL_COACH_PORT="4806",
                        CALL_COACH_CONFIG=str(self.tmp / "config.json"), CALL_COACH_NO_OPEN="1",
                        CALL_COACH_LOG=str(self.tmp / "launch.log"), CALL_COACH_ROOT=str(ROOT),
                        CALL_COACH_URL="http://127.0.0.1:4806/mcp", PATH=f"{FAKES}:{os.environ['PATH']}")

    def cli(self, *args, scope="owner"):
        env = dict(self.env, CALL_COACH_SCOPE=scope)
        out = subprocess.run([str(ROOT / ".venv" / "bin" / "call-coach-cli"), *args], env=env,
                             capture_output=True, text=True, timeout=60).stdout
        return json.loads(out)

    def wait_until_up(self):
        for _ in range(60):
            if subprocess.run(["curl", "-s", "-o", "/dev/null", "http://127.0.0.1:4806/"]).returncode == 0:
                return
            time.sleep(0.25)
        self.fail("Call Coach did not start")

    def saved_calls(self):
        index = self.tmp / "home" / "calls.json"
        return json.loads(index.read_text()) if index.exists() else []

    def test_closing_the_launcher_window_saves_the_call(self):
        pid, _ = pty.fork()
        if pid == 0:
            os.execve("/bin/bash", ["bash", str(ROOT / "scripts" / "Call Coach.command")], self.env)
        self.wait_until_up()
        self.cli("load_questions_file", f"path={self.tmp / 'questions.md'}")
        self.assertEqual(self.cli("start_call")["result"]["state"], "listening")
        os.killpg(os.getpgid(pid), signal.SIGHUP)
        for _ in range(80):
            if os.waitpid(pid, os.WNOHANG)[0]:
                break
            time.sleep(0.25)
        time.sleep(1)
        self.assertEqual(len(self.saved_calls()), 1)

    def test_a_term_signal_saves_the_call(self):
        proc = subprocess.Popen([str(ROOT / ".venv" / "bin" / "call-coach")], env=self.env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.wait_until_up()
        self.cli("load_questions_file", f"path={self.tmp / 'questions.md'}")
        self.cli("start_call")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
        self.assertEqual(len(self.saved_calls()), 1)

    def test_a_coaching_token_cannot_use_owner_tools_over_mcp(self):
        proc = subprocess.Popen([str(ROOT / ".venv" / "bin" / "call-coach")], env=self.env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.wait_until_up()
            refused = self.cli("load_questions_file", f"path={self.tmp / 'questions.md'}", scope="coach")
            self.assertFalse(refused["ok"])
            self.assertIn("owner", refused["error"]["message"])
            self.cli("load_questions_file", f"path={self.tmp / 'questions.md'}")
            self.assertFalse(self.cli("start_call", scope="coach")["ok"])
            self.assertEqual(self.cli("start_call")["result"]["state"], "listening")
            self.assertFalse(self.cli("stop_call", scope="coach")["ok"])
            self.assertFalse(self.cli("mark_question", "key=1", scope="coach")["ok"])
            self.assertTrue(self.cli("get_session", scope="coach")["ok"])
            self.assertEqual(self.cli("stop_call")["result"]["state"], "saved")
        finally:
            proc.terminate()
            proc.wait(timeout=60)


if __name__ == "__main__":
    unittest.main()
