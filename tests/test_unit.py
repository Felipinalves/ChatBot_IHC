from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from auth.auth_utils import login_with_email, register_user
from auth.pyrebase_auth import FirebaseAuth
from chat import chat_interface as ci
from rag.gemini_integration import GeminiError, redact, stream_with_fallback, to_user_error
from rag.rag_engine import Retriever
from tests.fakes import FakeAuth, FakeFirestore
from utils.time_utils import get_brasilia_time

LEAKED_KEY = "AIzaSyA0000000000000000000000000000000"


def make_retriever():
    vocab = ["usabilidade", "acessibilidade", "heurística"]

    def encode(text):
        vec = np.array([text.lower().count(w) for w in vocab], dtype=np.float32) + 1e-6
        return vec / np.linalg.norm(vec)

    chunks = [
        {"file_name": "Usabilidade em apps.txt", "text": "usabilidade usabilidade"},
        {"file_name": "Acessibilidade web.txt", "text": "acessibilidade na web"},
        {"file_name": "Heurísticas de Nielsen.txt", "text": "heurística de avaliação"},
    ]
    return Retriever(np.stack([encode(c["text"]) for c in chunks]), chunks, encode)


def test_retrieve_returns_most_similar_chunk_first():
    results = make_retriever().retrieve("o que é usabilidade?", top_k=3, min_score=0.5)
    assert [r.file_name for r in results] == ["Usabilidade em apps.txt"]


def test_retrieve_can_be_restricted_to_one_file():
    retriever = make_retriever()
    results = retriever.retrieve("usabilidade", min_score=0.0, file_name="Acessibilidade web.txt")
    assert {r.file_name for r in results} == {"Acessibilidade web.txt"}


def test_find_file_ignores_case_and_extension():
    retriever = make_retriever()
    assert retriever.find_file("heurísticas de nielsen") == "Heurísticas de Nielsen.txt"
    assert retriever.find_file("acessibilidade") == "Acessibilidade web.txt"
    assert retriever.find_file("inexistente") is None


def test_api_key_is_never_shown_to_the_user():
    exc = Exception(
        f"403 Permission denied: Consumer 'api_key:{LEAKED_KEY}' has been suspended. CONSUMER_SUSPENDED"
    )
    exc.code = 403
    error = to_user_error(exc)
    assert LEAKED_KEY not in error.user_message
    assert "chave da API" in error.user_message
    assert not error.retryable
    assert LEAKED_KEY not in redact(str(exc))


@pytest.mark.parametrize(
    "code,retryable,fallback",
    [(429, False, True), (503, True, True), (404, False, True), (400, False, False), (403, False, False)],
)
def test_error_mapping(code, retryable, fallback):
    exc = Exception("erro")
    exc.code = code
    error = to_user_error(exc)
    assert (error.retryable, error.fallback) == (retryable, fallback)


def test_aq_keys_are_redacted():
    assert "AQ.Ab8RN6" not in redact("key=AQ.Ab8RN6Kx_1234567890abcdefghij")


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} erro")
        self.code = code


class FakeGeminiClient:
    """Cada modelo devolve uma lista de resultados por tentativa: código de erro ou texto."""

    def __init__(self, script):
        self.script = {m: list(r) for m, r in script.items()}
        self.calls = []
        self.models = self

    def generate_content_stream(self, model, contents, config):
        self.calls.append(model)
        result = self.script[model].pop(0)
        if isinstance(result, int):
            raise ApiError(result)
        return [SimpleNamespace(text=part) for part in result]


def run_stream(client, models):
    return "".join(stream_with_fallback(client, models, "p", None, sleep=lambda s: None))


def test_overloaded_model_falls_back_to_next():
    client = FakeGeminiClient({"a": [503, 503], "b": [["resposta ", "ok"]]})
    assert run_stream(client, ["a", "b"]) == "resposta ok"
    assert client.calls == ["a", "a", "b"]


def test_transient_error_is_retried_on_same_model():
    client = FakeGeminiClient({"a": [503, ["ok"]], "b": []})
    assert run_stream(client, ["a", "b"]) == "ok"
    assert client.calls == ["a", "a"]


def test_quota_error_skips_retry_and_uses_fallback():
    client = FakeGeminiClient({"a": [429], "b": [["ok"]]})
    assert run_stream(client, ["a", "b"]) == "ok"
    assert client.calls == ["a", "b"]


def test_permission_error_does_not_fall_back():
    client = FakeGeminiClient({"a": [403], "b": [["ok"]]})
    with pytest.raises(GeminiError) as info:
        run_stream(client, ["a", "b"])
    assert "chave da API" in info.value.user_message
    assert client.calls == ["a"]


def test_all_models_overloaded_raises_friendly_error():
    client = FakeGeminiClient({"a": [503, 503], "b": [503, 503]})
    with pytest.raises(GeminiError) as info:
        run_stream(client, ["a", "b"])
    assert "sobrecarregado" in info.value.user_message


def test_build_prompt_includes_context_history_and_question():
    chunk = SimpleNamespace(file_name="a.txt", score=0.9, text="trecho relevante")
    history = [{"role": "user", "content": "pergunta anterior"}, {"role": "assistant", "content": "resposta"}]
    prompt = ci.build_prompt("nova pergunta", [chunk], history)
    assert "trecho relevante" in prompt and "pergunta anterior" in prompt and "nova pergunta" in prompt


def test_chat_title_is_truncated():
    assert ci.make_chat_title("curta") == "curta"
    title = ci.make_chat_title("x" * 80)
    assert len(title) == ci.MAX_TITLE_CHARS and title.endswith("...")


def test_firestore_round_trip_and_cleanup():
    db = FakeFirestore()
    chat_id = ci.create_new_chat_in_firestore(db, "u1", "Chat")
    messages = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "olá", "sources": ["a.txt"]},
    ]
    ci.save_messages_to_firestore(db, chat_id, messages)
    loaded = ci.load_chat_messages_from_firestore(db, chat_id)
    assert [m["content"] for m in loaded] == ["oi", "olá"]
    assert loaded[1]["sources"] == ["a.txt"]
    assert list(ci.list_user_chats_from_firestore(db, "u1")) == [chat_id]

    # Chat inativo há mais de 30 dias é removido junto com as mensagens
    db.docs[("chats", chat_id)]["updated_at"] = get_brasilia_time() - timedelta(days=31)
    ci.cleanup_old_chats(db)
    assert db.docs == {}


def test_login_error_is_translated():
    user, error = login_with_email(FakeAuth(), "aluno@utfpr.edu.br", "errada")
    assert user is None and "incorretos" in error


class FakeResponse:
    def __init__(self, status, data):
        self.status_code = status
        self.ok = status < 400
        self._data = data
        self.content = b"{}"

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, params, json, timeout):
        self.calls.append((url, params, json))
        return self.response


def test_firebase_rest_client_success_and_error():
    ok = FakeSession(FakeResponse(200, {"localId": "u1", "idToken": "t"}))
    user = FirebaseAuth("key", session=ok).sign_in_with_email_and_password("a@b.c", "x")
    assert user["localId"] == "u1"
    url, params, payload = ok.calls[0]
    assert url.endswith("accounts:signInWithPassword") and params == {"key": "key"}
    assert payload["returnSecureToken"] is True

    failing = FakeSession(FakeResponse(400, {"error": {"message": "EMAIL_EXISTS"}}))
    user, error = register_user(FirebaseAuth("key", session=failing), FakeFirestore(), "a@b.c", "x", "A")
    assert user is None and "já possui uma conta" in error

    weak = FakeSession(FakeResponse(400, {"error": {"message": "WEAK_PASSWORD : Password should be at least 6 characters"}}))
    _, error = register_user(FirebaseAuth("key", session=weak), FakeFirestore(), "a@b.c", "x", "A")
    assert "6 caracteres" in error


def test_login_handles_unexpected_errors():
    class BrokenAuth:
        def sign_in_with_email_and_password(self, *_):
            raise ConnectionError("sem rede")

    user, error = login_with_email(BrokenAuth(), "a@b.c", "x")
    assert user is None and error
