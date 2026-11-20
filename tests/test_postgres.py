"""Integration test against a real PostgreSQL. Skipped unless LOOPGUARD_TEST_PG_URL is set.

export LOOPGUARD_TEST_PG_URL=postgresql://postgres@localhost:5432/loopguard
pytest tests/test_postgres.py
"""

import os
import threading

import pytest

from loopguard.agent.graph import ReviewDecision
from loopguard.config import Settings
from loopguard.service import AgentService, RunConflict

PG_URL = os.getenv("LOOPGUARD_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="LOOPGUARD_TEST_PG_URL not set")


def _service() -> AgentService:
    return AgentService(Settings(database_url=PG_URL, checkpoint_url=PG_URL, log_json=False))


def test_replicas_start_together_and_share_runs():
    services: list[AgentService] = []
    errors: list[Exception] = []

    def boot() -> None:
        try:
            services.append(_service())
        except Exception as exc:  # pragma: no cover - this is what the test guards against
            errors.append(exc)

    threads = [threading.Thread(target=boot) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(services) == 3

    a, b, c = services
    run = a.start_run("What's the weather in Kolkata right now?")
    assert run["status"] == "paused"
    done = b.resume_run(run["run_id"], ReviewDecision(action="approve", reviewer="pg-test"))
    assert done["status"] == "completed"
    with pytest.raises(RunConflict):
        c.resume_run(run["run_id"], ReviewDecision(action="approve"))
    assert c.get_run(run["run_id"])["status"] == "completed"
