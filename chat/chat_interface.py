import re
from datetime import timedelta

import streamlit as st

from rag.gemini_integration import GeminiError, stream_response
from utils.time_utils import get_brasilia_time

CHAT_RETENTION_DAYS = 30
HISTORY_TURNS = 6  # mensagens anteriores enviadas como contexto da conversa
MAX_CHUNK_CHARS = 1500
MAX_TITLE_CHARS = 50

ARTICLE_PATTERN = re.compile(r"artigo:\s*[\"'](.+?)[\"']", re.IGNORECASE)

SYSTEM_INSTRUCTION = """Você é um especialista em IHC (Interação Humano-Computador) com vasta experiência acadêmica e prática.

[INSTRUÇÕES]
1. Analise cuidadosamente a pergunta, o histórico da conversa e o contexto fornecido.
2. Se o contexto contiver informações relevantes para a pergunta, baseie sua resposta principalmente nessas informações.
3. Se o contexto for insuficiente ou não abordar diretamente a pergunta, responda com base em seu conhecimento geral de IHC, sem mencionar a ausência de informações no contexto.
4. Não faça referências diretas aos "textos fornecidos", "trechos" ou "artigos" na sua resposta.
5. Se a pergunta não tiver relação com IHC, explique educadamente que você é especializado em IHC.

[FORMATO]
- Use português brasileiro formal
- Mantenha termos técnicos consolidados em inglês quando apropriado
- Estruture a resposta em parágrafos claros e concisos, usando listas e Markdown quando ajudarem a leitura
- Inclua exemplos práticos quando relevante
- Apresente diferentes perspectivas quando apropriado"""

EXAMPLE_QUESTIONS = [
    ("Heurísticas de Nielsen", "O que são as heurísticas de Nielsen?"),
    ("Testes de usabilidade", "Como funciona a avaliação de usabilidade com usuários?"),
    ("Acessibilidade na web", "Quais cuidados de acessibilidade devo ter em interfaces web?"),
]

AVATARS = {"user": ":material/person:", "assistant": ":material/smart_toy:"}


def build_prompt(question, chunks, history):
    context = "\n\n".join(
        f"[Fonte: {c.file_name} | relevância {c.score:.2f}]\n{c.text[:MAX_CHUNK_CHARS]}"
        for c in chunks
    ) or "(nenhum trecho relevante encontrado)"

    recent = history[-HISTORY_TURNS:]
    conversation = "\n".join(
        f"{'Usuário' if m['role'] == 'user' else 'Assistente'}: {m['content']}" for m in recent
    ) or "(início da conversa)"

    return f"""[CONTEXTO]
{context}

[HISTÓRICO DA CONVERSA]
{conversation}

[PERGUNTA]
{question}"""


def make_chat_title(prompt):
    title = " ".join(prompt.split())
    if len(title) > MAX_TITLE_CHARS:
        title = title[: MAX_TITLE_CHARS - 3].rstrip() + "..."
    return title


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Fontes consultadas ({len(sources)})", icon=":material/menu_book:"):
        for name in sources:
            st.markdown(f"- {name.removesuffix('.txt')}")


def render_message(message):
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources"))


def show_chat_interface(retriever, firestore_db, user_id):
    """Exibe a interface principal do chatbot."""
    messages = st.session_state.messages

    if not messages:
        st.markdown("### Olá! Como posso ajudar nos seus estudos de IHC?")
        st.caption(
            "As respostas são baseadas em um acervo de artigos científicos da área. "
            'Para perguntar sobre um artigo específico, use: artigo: "nome do artigo".'
        )
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, (label, question) in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(label, help=question, use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()

    for message in messages:
        render_message(message)

    prompt = st.chat_input("Faça uma pergunta sobre IHC")
    prompt = prompt or st.session_state.pop("pending_prompt", None)
    if not prompt:
        return

    if retriever is None:
        st.error("A base de conhecimento não está disponível no momento. Tente novamente mais tarde.")
        return

    user_message = {"role": "user", "content": prompt, "timestamp": get_brasilia_time()}
    render_message(user_message)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Buscando informações relevantes..."):
            file_name = None
            match = ARTICLE_PATTERN.search(prompt)
            if match:
                file_name = retriever.find_file(match.group(1))
                if file_name is None:
                    st.warning(f'Artigo "{match.group(1)}" não encontrado. Buscando em todo o acervo.')
            if file_name:
                chunks = retriever.retrieve(prompt, top_k=10, min_score=0.0, file_name=file_name)
            else:
                chunks = retriever.retrieve(prompt)

        full_prompt = build_prompt(prompt, chunks, messages)
        try:
            response = st.write_stream(stream_response(full_prompt, SYSTEM_INSTRUCTION))
        except GeminiError as e:
            st.error(e.user_message)
            return

        if isinstance(response, list):
            response = "".join(str(part) for part in response)
        response = (response or "").strip()
        if not response:
            st.error("O modelo não retornou uma resposta. Tente reformular a pergunta.")
            return

        sources = list(dict.fromkeys(c.file_name for c in chunks))
        render_sources(sources)

    assistant_message = {
        "role": "assistant",
        "content": response,
        "sources": sources,
        "timestamp": get_brasilia_time(),
    }

    # O chat só é criado no Firestore quando a primeira pergunta é respondida
    if not st.session_state.get("current_chat_id"):
        title = make_chat_title(prompt)
        chat_id = create_new_chat_in_firestore(firestore_db, user_id, title)
        st.session_state.current_chat_id = chat_id
        st.session_state.chats = {chat_id: {"title": title}, **st.session_state.get("chats", {})}

    messages.extend([user_message, assistant_message])
    save_messages_to_firestore(
        firestore_db, st.session_state.current_chat_id, messages, start=len(messages) - 2
    )
    st.rerun()


def save_messages_to_firestore(firestore_db, chat_id, messages, start=0):
    """Salva as mensagens a partir do índice `start` e atualiza a data do chat."""
    now = get_brasilia_time()
    chat_ref = firestore_db.collection("chats").document(chat_id)
    batch = firestore_db.batch()
    for order, msg in enumerate(messages[start:], start=start):
        data = {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg.get("timestamp", now),
            "order": order,
        }
        if msg.get("sources"):
            data["sources"] = msg["sources"]
        batch.set(chat_ref.collection("messages").document(), data)
    batch.update(chat_ref, {"updated_at": now, "expiry_date": now + timedelta(days=CHAT_RETENTION_DAYS)})
    batch.commit()


def load_chat_messages_from_firestore(firestore_db, chat_id):
    """Carrega as mensagens de um chat, em ordem."""
    if not chat_id:
        return []
    refs = (
        firestore_db.collection("chats")
        .document(chat_id)
        .collection("messages")
        .order_by("order")
        .stream()
    )
    messages = []
    for ref in refs:
        data = ref.to_dict()
        messages.append(
            {
                "role": data["role"],
                "content": data["content"],
                "sources": data.get("sources", []),
                "timestamp": data.get("timestamp"),
            }
        )
    return messages


def create_new_chat_in_firestore(firestore_db, user_id, title):
    now = get_brasilia_time()
    chat_ref = firestore_db.collection("chats").document()
    chat_ref.set(
        {
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "expiry_date": now + timedelta(days=CHAT_RETENTION_DAYS),
        }
    )
    return chat_ref.id


def list_user_chats_from_firestore(firestore_db, user_id, limit=50):
    """Lista os chats do usuário, do mais recente para o mais antigo."""
    query = (
        firestore_db.collection("chats")
        .where("user_id", "==", user_id)
        .order_by("updated_at", direction="DESCENDING")
        .limit(limit)
    )
    return {chat.id: chat.to_dict() for chat in query.stream()}


def delete_chat_from_firestore(firestore_db, chat_id):
    chat_ref = firestore_db.collection("chats").document(chat_id)
    _delete_chats(firestore_db, [chat_ref])


def cleanup_old_chats(firestore_db, days=CHAT_RETENTION_DAYS):
    """Remove chats (e suas mensagens) sem atividade há mais de `days` dias."""
    cutoff = get_brasilia_time() - timedelta(days=days)
    expired = firestore_db.collection("chats").where("updated_at", "<", cutoff).stream()
    count = _delete_chats(firestore_db, [chat.reference for chat in expired])
    print(f"Limpeza de chats antigos concluída: {count} chats removidos")


def _delete_chats(firestore_db, chat_refs):
    batch, ops, count = firestore_db.batch(), 0, 0
    for chat_ref in chat_refs:
        for msg in chat_ref.collection("messages").stream():
            batch.delete(msg.reference)
            ops += 1
            if ops >= 450:  # o Firestore aceita no máximo 500 operações por batch
                batch.commit()
                batch, ops = firestore_db.batch(), 0
        batch.delete(chat_ref)
        ops += 1
        count += 1
        if ops >= 450:
            batch.commit()
            batch, ops = firestore_db.batch(), 0
    if ops:
        batch.commit()
    return count
