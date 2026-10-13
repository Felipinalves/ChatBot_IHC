"""Testes de interface com o AppTest do Streamlit, usando Firebase e Gemini falsos."""
import pytest
from streamlit.testing.v1 import AppTest

import auth.firebase_auth
import auth.pyrebase_auth
import chat.chat_interface
import rag.rag_engine
from rag.gemini_integration import GeminiError
from tests.fakes import FakeFirebaseApp, FakeFirestore
from tests.test_unit import make_retriever

DB = FakeFirestore()


def fake_stream(prompt, system_instruction=None):
    yield "As heurísticas de Nielsen "
    yield "são dez princípios de usabilidade."


def failing_stream(prompt, system_instruction=None):
    raise GeminiError("O serviço de IA está indisponível: a chave da API do Gemini é inválida.")
    yield  # pragma: no cover


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    DB.docs.clear()
    firebase_app = FakeFirebaseApp()
    monkeypatch.setattr(auth.firebase_auth, "initialize_firebase_admin", lambda: DB)
    monkeypatch.setattr(auth.pyrebase_auth, "initialize_pyrebase", lambda: firebase_app)
    monkeypatch.setattr(rag.rag_engine, "initialize_system", make_retriever)
    monkeypatch.setattr(chat.chat_interface, "stream_response", fake_stream)


def run_app():
    at = AppTest.from_file("../streamlit_app.py", default_timeout=60)
    return at.run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def login(at):
    at.text_input(key="login_email").input("aluno@utfpr.edu.br")
    at.text_input(key="login_password").input("senha123")
    return button(at, "Entrar").click().run()


def test_login_page_shows_utfpr_logo():
    at = run_app()
    assert not at.exception
    html = "".join(m.value for m in at.markdown)
    assert "utfpr-logo" in html and "Logo da UTFPR" in html
    assert [t.label for t in at.tabs] == ["Entrar", "Criar conta", "Recuperar senha"]


def test_wrong_password_shows_error():
    at = run_app()
    at.text_input(key="login_email").input("aluno@utfpr.edu.br")
    at.text_input(key="login_password").input("errada")
    at = button(at, "Entrar").click().run()
    assert any("incorretos" in e.value for e in at.error)


def test_chat_flow_creates_chat_only_after_first_answer():
    at = login(run_app())
    assert not at.exception
    assert at.session_state.user_id == "user-1"
    assert not [k for k in DB.docs if k[0] == "chats"]  # nenhum chat vazio criado

    at.chat_input[0].set_value('o que é usabilidade? artigo: "Usabilidade em apps"').run()
    assert not at.exception
    chats = [k for k in DB.docs if k[0] == "chats"]
    assert len(chats) == 1
    assert DB.docs[chats[0]]["title"].startswith("o que é usabilidade?")
    messages = at.session_state.messages
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert "dez princípios" in messages[1]["content"]
    assert messages[1]["sources"] == ["Usabilidade em apps.txt"]

    # O chat aparece no menu e pode ser excluído
    html = "".join(m.value for m in at.sidebar.markdown)
    assert "chat-current" in html and "o que é usabilidade?" in html
    at = button(at, "Excluir chat atual").click().run()
    assert not [k for k in DB.docs if k[0] == "chats"]
    assert at.session_state.messages == []


def test_gemini_error_is_shown_without_crashing(monkeypatch):
    monkeypatch.setattr(chat.chat_interface, "stream_response", failing_stream)
    at = login(run_app())
    at.chat_input[0].set_value("teste").run()
    assert not at.exception
    assert any("chave da API" in e.value for e in at.error)
    assert at.session_state.messages == []
