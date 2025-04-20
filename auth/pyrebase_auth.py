import pyrebase
import streamlit as st

@st.cache_resource(show_spinner=False)
def initialize_pyrebase():
    try:
        firebase_config = {
            "apiKey": st.secrets["FIREBASE"]["apiKey"],
            "authDomain": st.secrets["FIREBASE"]["authDomain"],
            "projectId": st.secrets["FIREBASE"]["projectId"],
            "storageBucket": st.secrets["FIREBASE"]["storageBucket"],
            "messagingSenderId": st.secrets["FIREBASE"]["messagingSenderId"],
            "appId": st.secrets["FIREBASE"]["appId"],
            "databaseURL": st.secrets["FIREBASE"].get("databaseURL", "")  # Usando get para evitar KeyError
        }
        firebase = pyrebase.initialize_app(firebase_config)
        return firebase
    except Exception as e:
        st.error(f"Erro ao inicializar Pyrebase: {str(e)}")
        return None

# Exemplo de uso para histórico de chat
def save_chat_history(db, user_id, message, is_bot=False):
    """
    Salva uma mensagem no histórico de chat do usuário no Firestore
    
    Args:
        db: Cliente Firestore
        user_id: ID do usuário
        message: Texto da mensagem
        is_bot: Se a mensagem é do bot (True) ou do usuário (False)
    """
    # Referência para a coleção de chats do usuário
    chat_ref = db.collection("chats").document(user_id)
    
    # Verifica se o documento do chat existe
    chat_doc = chat_ref.get()
    
    # Se não existir, cria o documento do chat
    if not chat_doc.exists:
        chat_ref.set({
            "user_id": user_id,
            "created_at": get_brasilia_time(),
            "updated_at": get_brasilia_time()
        })
    else:
        # Atualiza o timestamp do último update
        chat_ref.update({"updated_at": get_brasilia_time()})
    
    # Adiciona a mensagem à subcoleção de mensagens
    chat_ref.collection("messages").add({
        "sender": "bot" if is_bot else "user",
        "content": message,
        "timestamp": get_brasilia_time()
    })

def get_chat_history(db, user_id, limit=50):
    """
    Obtém o histórico de chat do usuário do Firestore
    
    Args:
        db: Cliente Firestore
        user_id: ID do usuário
        limit: Número máximo de mensagens a retornar
        
    Returns:
        Uma lista de mensagens ordenadas por timestamp
    """
    messages = []
    
    # Obtém as mensagens da subcoleção, ordenadas por timestamp
    message_refs = db.collection("chats").document(user_id) \
                     .collection("messages") \
                     .order_by("timestamp") \
                     .limit(limit) \
                     .stream()
    
    # Converte os documentos para dicionários e adiciona à lista
    for msg in message_refs:
        msg_data = msg.to_dict()
        messages.append(msg_data)
    
    return messages
