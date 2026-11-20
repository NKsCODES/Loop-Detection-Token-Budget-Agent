import pytest
from fastapi.testclient import TestClient

from loopguard.api.app import create_app
from loopguard.config import Settings
from loopguard.service import AgentService


@pytest.fixture
def client():
    settings = Settings(
        database_url="sqlite://", checkpoint_url="memory", log_json=False, api_key="secret"
    )
    app = create_app(settings, AgentService(settings))
    with TestClient(app, headers={"X-API-Key": "secret"}) as c:
        yield c


def test_probes(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["status"] == "ready"
    assert "loopguard_runs_started_total" in client.get("/metrics").text


def test_auth_required(client):
    r = client.get("/v1/runs", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_pause_review_resume_flow(client):
    r = client.post("/v1/runs", json={"task": "What's the weather in Kolkata right now?"})
    assert r.status_code == 201
    run = r.json()
    assert run["status"] == "paused" and run["pending"]["detector"] == "exact_repeat"
    queue = client.get("/v1/runs", params={"status": "paused"}).json()
    assert [q["run_id"] for q in queue] == [run["run_id"]]

    bad = client.post(f"/v1/runs/{run['run_id']}/decision", json={"action": "extend_budget"})
    assert bad.status_code == 409

    done = client.post(
        f"/v1/runs/{run['run_id']}/decision", json={"action": "approve", "reviewer": "qa"}
    ).json()
    assert done["status"] == "completed" and "unavailable" in done["final_answer"]
    assert done["guard_events"][0]["reviewer"] == "qa"

    again = client.post(f"/v1/runs/{run['run_id']}/decision", json={"action": "approve"})
    assert again.status_code == 409  # cannot resume a finished run

    metrics = client.get("/metrics").text
    assert 'loopguard_hitl_pauses_total{detector="exact_repeat",kind="loop"}' in metrics


def test_guard_overrides_validated(client):
    r = client.post("/v1/runs", json={"task": "x", "guard": {"max_steps": 0}})
    assert r.status_code == 422
    r = client.post("/v1/runs", json={"task": "x", "guard": {"unknown": 1}})
    assert r.status_code == 422


def test_override_abort_mode(client):
    r = client.post(
        "/v1/runs",
        json={
            "task": "How many open tickets does project legacy-hr have?",
            "guard": {"on_loop": "abort"},
        },
    ).json()
    assert r["status"] == "aborted" and r["stop_reason"] == "loop"


def test_not_found(client):
    assert client.get("/v1/runs/does-not-exist").status_code == 404


def test_console_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "LoopGuard" in r.text
