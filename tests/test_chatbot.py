"""Offline tests - they use the MockProvider, so no API key or internet is needed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app import create_app
from chat_core import ChatService
from providers import LLMError, MockProvider


def make_service(strategy="summary", max_messages=6, keep_recent=2):
    return ChatService(MockProvider(), strategy=strategy, max_messages=max_messages, keep_recent=keep_recent)


def test_model_sees_earlier_messages():
    svc = make_service()
    first = svc.chat(None, "My name is Yukthi")
    cid = first["conversation_id"]
    svc.chat(cid, "I like Python")
    svc.chat(cid, "I want a data engineering job")
    reply = svc.chat(cid, "what did I say first?")["reply"]
    assert "My name is Yukthi" in reply  # first message is still in context 3 turns later


def test_system_prompt_is_sent():
    svc = make_service()
    svc.chat(None, "hello")
    system, _ = svc.provider.calls[-1]
    assert "Sage" in system


def test_conversations_are_isolated():
    svc = make_service()
    a = svc.chat(None, "alpha")["conversation_id"]
    b = svc.chat(None, "beta")["conversation_id"]
    assert a != b
    svc.chat(a, "again")
    _, msgs = svc.provider.calls[-1]
    assert all("beta" not in m["content"] for m in msgs)


def test_truncate_strategy_drops_old_messages():
    svc = make_service(strategy="truncate", max_messages=6, keep_recent=2)
    cid = None
    for i in range(8):
        cid = svc.chat(cid, f"message {i}")["conversation_id"]
    _, msgs = svc.provider.calls[-1]
    assert len(msgs) <= 6 + 1
    assert msgs[0]["role"] == "user"
    assert all("message 0" not in m["content"] for m in msgs)


def test_summary_strategy_prepends_summary_and_bounds_history():
    svc = make_service(strategy="summary", max_messages=6, keep_recent=2)
    cid = None
    for i in range(8):
        result = svc.chat(cid, f"message {i}")
        cid = result["conversation_id"]
    system, msgs = svc.provider.calls[-1]
    assert "Summary of the earlier part" in system
    assert len(msgs) <= 6 + 1
    assert result["context"]["has_summary"] is True


def test_failed_call_is_rolled_back():
    svc = make_service()
    cid = svc.chat(None, "hello")["conversation_id"]
    with pytest.raises(LLMError):
        svc.chat(cid, "please FAIL")
    convo = svc.store.get_or_create(cid)
    assert [m["content"] for m in convo.messages] == ["hello", "[mock] You said: hello (I can see 1 messages of context)"]


def test_empty_and_oversized_messages_rejected():
    svc = make_service()
    with pytest.raises(ValueError):
        svc.chat(None, "   ")
    with pytest.raises(ValueError):
        svc.chat(None, "x" * 5000)


# ---------------------------------------------------------------- HTTP layer
@pytest.fixture
def client():
    app = create_app(make_service())
    return app.test_client()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_chat_endpoint_multi_turn(client):
    r1 = client.post("/api/chat", json={"message": "I am learning SQL"})
    cid = r1.get_json()["conversation_id"]
    r2 = client.post("/api/chat", json={"conversation_id": cid, "message": "what did I say first?"})
    assert r2.status_code == 200
    assert "I am learning SQL" in r2.get_json()["reply"]


def test_chat_endpoint_validation(client):
    assert client.post("/api/chat", data="not json").status_code == 400
    assert client.post("/api/chat", json={"message": ""}).status_code == 400


def test_api_failure_returns_clear_json_error(client):
    r = client.post("/api/chat", json={"message": "FAIL"})
    assert r.status_code == 502
    assert "error" in r.get_json() and "Traceback" not in r.get_json()["error"]


def test_timeout_maps_to_504(client):
    r = client.post("/api/chat", json={"message": "SLOW"})
    assert r.status_code == 504
    assert r.get_json()["kind"] == "timeout"


def test_reset_clears_history(client):
    cid = client.post("/api/chat", json={"message": "remember pineapple"}).get_json()["conversation_id"]
    assert client.delete(f"/api/conversations/{cid}").status_code == 200
    reply = client.post("/api/chat", json={"conversation_id": cid, "message": "what did I say first?"}).get_json()["reply"]
    assert "pineapple" not in reply
