import os
import tempfile

# Tests never touch your real calls, token or settings.
os.environ["CALL_COACH_HOME"] = tempfile.mkdtemp(prefix="call-coach-test-")
os.environ.pop("CALL_COACH_CALLS_INDEX", None)
os.environ.pop("TYPESAFE_API_KEY", None)
