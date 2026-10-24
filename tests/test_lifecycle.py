"""Start, listen, fail and stop with stand-ins for the audio helper and whisper-server (tests/fakes)."""

import asyncio
import json
import os
import socket
import unittest
from pathlib import Path

from callcoach import app, store

FAKES = Path(__file__).parent / "fakes"
QUESTIONS = """# Lifecycle test

- When did you start?
- Who was first?
"""


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = dict(os.environ)
        os.environ["PATH"] = f"{FAKES}{os.pathsep}{os.environ['PATH']}"
        for key in ("FAKE_CAPTURE_MODE", "FAKE_WHISPER_MODE", "CALL_COACH_WHISPER_PORT", "CALL_COACH_NO_MIC"):
            os.environ.pop(key, None)
        folder = Path(os.environ["CALL_COACH_HOME"]) / f"q-{self.id().split('.')[-1]}"
        folder.mkdir(parents=True, exist_ok=True)
        self.questions = folder / "questions.md"
        self.questions.write_text(QUESTIONS)
        model = store.SUPPORT / "models" / "ggml-small.en.bin"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"fake")
        app.session = app.Session()
        app.config.clear()
        app.config.update({"capture_bin": str(FAKES / "cc-capture"), "jev": {"api_key_env": "CC_NO_SUCH_KEY"}})
        app.jev = app.pipeline = None
        app.load_questions(self.questions)

    async def asyncTearDown(self):
        if app.session.running:
            await app.stop_session()
        os.environ.clear()
        os.environ.update(self.env)

    async def test_start_listens_only_once_audio_really_works_then_stop_saves(self):
        os.environ["FAKE_WHISPER_TEXT"] = " Hello there, this is a test."
        await app.start_session()
        self.assertEqual(app.session.state, "listening")
        self.assertEqual(app.session.health["jev"]["state"], "off")
        await asyncio.sleep(3.5)  # the fake speaks for 1.5 s, then pauses
        await app.stop_session()
        self.assertEqual(app.session.state, "saved")
        texts = [l["text"] for l in app.session.lines]
        self.assertIn("Hello there, this is a test.", texts)
        folder = store.call_dir(app.session.call_id)
        self.assertIn("Hello there", (folder / "transcript.md").read_text())
        self.assertTrue(Path(app.session.saved[0]).exists())
        self.assertEqual(store.read_index()[0]["id"], app.session.call_id)
        self.assertEqual(oct(folder.stat().st_mode & 0o777), "0o700")

    async def test_speech_still_buffered_at_stop_is_transcribed(self):
        os.environ["FAKE_SPEECH_SECONDS"] = "30"
        await app.start_session()
        await asyncio.sleep(2.0)  # mid-sentence: no pause yet
        await app.stop_session()
        self.assertTrue(any(l["speaker"] == "You" for l in app.session.lines))

    async def test_missing_screen_recording_permission_fails_clearly(self):
        os.environ["FAKE_CAPTURE_MODE"] = "deny"
        with self.assertRaises(RuntimeError) as caught:
            await app.start_session()
        self.assertIn("Screen & System Audio Recording", str(caught.exception))
        self.assertEqual(app.session.state, "failed")
        self.assertIsNone(app.pipeline)

    async def test_capture_dying_mid_call_is_reported_and_the_transcript_kept(self):
        os.environ["FAKE_CAPTURE_MODE"] = "die"
        await app.start_session()
        for _ in range(60):
            await asyncio.sleep(0.1)
            if app.session.health.get("capture", {}).get("state") == "failed":
                break
        self.assertEqual(app.session.health["capture"]["state"], "failed")
        self.assertEqual(app.session.severity, "error")
        await app.stop_session()
        self.assertEqual(app.session.state, "saved")

    async def test_capturing_without_any_frames_is_not_listening(self):
        os.environ["FAKE_CAPTURE_MODE"] = "no_frames"
        with self.assertRaises(RuntimeError) as caught:
            await app.start_session()
        self.assertIn("No audio arrived", str(caught.exception))
        self.assertEqual(app.session.state, "failed")

    async def test_a_silent_microphone_is_reported(self):
        os.environ["FAKE_CAPTURE_MODE"] = "silent_mic"
        await app.start_session()
        for _ in range(90):
            await asyncio.sleep(0.1)
            if app.session.health.get("capture", {}).get("state") == "degraded":
                break
        self.assertIn("microphone", app.session.health["capture"]["message"])

    async def test_whisper_port_already_taken_is_refused(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            os.environ["CALL_COACH_WHISPER_PORT"] = str(busy.getsockname()[1])
            with self.assertRaises(RuntimeError) as caught:
                await app.start_session()
        self.assertIn("already in use", str(caught.exception))

    async def test_whisper_that_cannot_start_is_reported(self):
        os.environ["FAKE_WHISPER_MODE"] = "crash"
        with self.assertRaises(RuntimeError) as caught:
            await app.start_session()
        self.assertIn("failed to load model", str(caught.exception))

    async def test_failed_transcription_is_recorded_not_dropped(self):
        os.environ["FAKE_WHISPER_MODE"] = "fail_inference"
        await app.start_session()
        await asyncio.sleep(3.5)
        await app.stop_session()
        self.assertTrue(any(l.get("failed") for l in app.session.lines))

    async def test_two_starts_at_once_make_one_call(self):
        await asyncio.gather(app.start_session(), app.start_session())
        self.assertEqual(app.session.state, "listening")
        pipeline = app.pipeline
        await app.start_session()
        self.assertIs(app.pipeline, pipeline)

    async def test_a_failed_save_can_be_retried(self):
        await app.start_session()
        real = store.write_atomic
        calls = {"n": 0}

        def flaky(path, text):
            if path.name == "transcript.md":
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("disk full")
            return real(path, text)
        store.write_atomic = flaky
        try:
            await app.stop_session()
            self.assertEqual(app.session.state, "save_failed")
            await app.stop_session()  # Stop again retries the save
        finally:
            store.write_atomic = real
        self.assertEqual(app.session.state, "saved")

    async def test_an_interrupted_call_is_saved_at_the_next_start(self):
        await app.start_session()
        await app.checkpoint()
        call_id = app.session.call_id
        app.session.state = "idle"  # pretend the process died without stopping
        await app.pipeline.stop(drain=False)
        app.pipeline = None
        self.assertIn(call_id, app.recover_interrupted())
        self.assertTrue(store.read_index()[0].get("recovered"))
        self.assertTrue((store.call_dir(call_id) / "results.md").exists())


class DoctorTests(unittest.TestCase):
    def test_doctor_reports_missing_permission(self):
        import contextlib
        import io
        from callcoach import doctor
        os.environ["FAKE_CAPTURE_MODE"] = "deny"
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = doctor.run({}, FAKES / "cc-capture")
        finally:
            os.environ.pop("FAKE_CAPTURE_MODE")
        self.assertEqual(code, 1)
        self.assertIn("Screen & System Audio Recording permission: not given", out.getvalue())


if __name__ == "__main__":
    unittest.main()
