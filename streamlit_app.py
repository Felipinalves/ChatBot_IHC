import html
import logging
import threading
import time
from datetime import timedelta

import streamlit as st

st.set_page_config(
    page_title="IAHC ChatBot – UTFPR",
    page_icon="assets/utfpr_simbolo.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

from auth.auth_interface import show_auth_page  # noqa: E402
from auth.firebase_auth import initialize_firebase_admin  # noqa: E402
from auth.pyrebase_auth import initialize_pyrebase  # noqa: E402
from chat.chat_interface import (  # noqa: E402
    cleanup_old_chats,
    delete_chat_from_firestore,
    list_user_chats_from_firestore,
    load_chat_messages_from_firestore,
    show_chat_interface,
)
from rag.rag_engine import initialize_system  # noqa: E402
from utils.branding import apply_brand_logo, apply_styles  # noqa: E402
from utils.time_utils import get_brasilia_time  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CLEANUP_HOUR = 3  # horário de Brasília


def _seconds_until_next_cleanup():
    now = get_brasilia_time()
    next_run = now.replace(hour=CLEANUP_HOUR, minute=0, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()


@st.cache_resource(show_spinner=False)
def start_cleanup_scheduler(_firestore_db):
    """Inicia uma única thread por servidor para remover chats inativos diariamente."""

    def run():
        while True:
            time.sleep(_seconds_until_next_cleanup())
            try:
                cleanup_old_chats(_firestore_db)
            except Exception:
                logger.exception("Falha na limpeza de chats antigos")

    thread = threading.Thread(target=run, name="cleanup-old-chats", daemon=True)
    thread.start()
    return thread


def start_new_chat():
    st.session_state.current_chat_id = None
    st.session_state.messages = []


def logout():
    st.session_state.clear()


def show_sidebar(firestore_db):
    with st.sidebar:
        with st.expander("Minha conta", icon=":material/account_circle:"):
            # Exibido como texto puro para o e-mail não virar um link
            st.text(st.session_state.get("user_email", ""))
            if st.button("Sair", icon=":material/logout:", use_container_width=True):
                logout()
                st.rerun()

        if st.button("Novo chat", icon=":material/add:", type="primary", use_container_width=True):
            start_new_chat()
            st.rerun()

        st.subheader("Meus chats")
        chats = st.session_state.chats
        if not chats:
            st.caption("Suas conversas aparecerão aqui.")

        current = st.session_state.get("current_chat_id")
        for chat_id, chat in chats.items():
            label = chat.get("title") or "Conversa sem título"
            if chat_id == current:
                st.markdown(f"<div class='chat-current'>{html.escape(label)}</div>", unsafe_allow_html=True)
            elif st.button(label, key=f"chat_{chat_id}", icon=":material/chat_bubble_outline:", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = load_chat_messages_from_firestore(firestore_db, chat_id)
                st.rerun()

        if current:
            st.divider()
            if st.button("Excluir chat atual", icon=":material/delete:", use_container_width=True):
                delete_chat_from_firestore(firestore_db, current)
                st.session_state.chats.pop(current, None)
                start_new_chat()
                st.rerun()

        st.caption("Conversas sem atividade por 30 dias são excluídas automaticamente.")


def main():
    apply_styles()

    firestore_db = initialize_firebase_admin()
    pyrebase_app = initialize_pyrebase()
    auth = pyrebase_app.auth() if pyrebase_app else None

    if firestore_db is not None:
        start_cleanup_scheduler(firestore_db)

    if not st.session_state.get("user_id"):
        show_auth_page(auth, firestore_db)
        return

    apply_brand_logo()

    if "chats" not in st.session_state:
        st.session_state.chats = list_user_chats_from_firestore(firestore_db, st.session_state.user_id)
    if "messages" not in st.session_state:
        start_new_chat()

    show_sidebar(firestore_db)

    st.title("IAHC ChatBot")
    st.caption("Assistente de estudos em Interação Humano-Computador • UTFPR")

    try:
        retriever = initialize_system()
    except Exception:
        logger.exception("Erro ao carregar a base de conhecimento")
        retriever = None

    show_chat_interface(retriever, firestore_db, st.session_state.user_id)


if __name__ == "__main__":
    main()
