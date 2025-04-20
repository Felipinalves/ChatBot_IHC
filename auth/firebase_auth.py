import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

def get_firebase_credentials():
    if "FIREBASE_SERVICE_ACCOUNT" in st.secrets:
        # Retorna diretamente o dicionário de credenciais
        return dict(st.secrets["FIREBASE_SERVICE_ACCOUNT"])
    else:
        raise Exception("Firebase credentials not found in Streamlit secrets.")

@st.cache_resource(show_spinner=False)
def initialize_firebase_admin():
    try:
        if not firebase_admin._apps:
            # Obtém as credenciais
            cred_dict = get_firebase_credentials()
            # Converte a chave privada de string para o formato correto
            if isinstance(cred_dict["private_key"], str):
                cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
            # Cria as credenciais do Firebase
            cred = credentials.Certificate(cred_dict)
            # Inicializa o Firebase Admin SDK
            firebase_admin.initialize_app(cred)
        # Retorna o cliente Firestore
        return firestore.client()
    except Exception as e:
        st.error(f"Erro ao inicializar Firebase Admin SDK: {str(e)}")
        return None
