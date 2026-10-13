"""Cliente mínimo da API REST do Firebase Authentication.

Substitui a biblioteca pyrebase4, que não é mais mantida e quebra com o urllib3 2.x.
Mantém a mesma interface usada pelo app (sign_in..., create_user..., send_password_reset_email).
"""
import logging

import requests
import streamlit as st

logger = logging.getLogger(__name__)

IDENTITY_TOOLKIT_URL = "https://identitytoolkit.googleapis.com/v1/accounts:{}"
TIMEOUT = 15


class FirebaseAuthError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class FirebaseAuth:
    def __init__(self, api_key, session=None):
        self._api_key = api_key
        self._session = session or requests.Session()

    def _post(self, method, payload):
        response = self._session.post(
            IDENTITY_TOOLKIT_URL.format(method),
            params={"key": self._api_key},
            json=payload,
            timeout=TIMEOUT,
        )
        data = response.json() if response.content else {}
        if not response.ok:
            raise FirebaseAuthError(data.get("error", {}).get("message", f"HTTP_{response.status_code}"))
        return data

    def sign_in_with_email_and_password(self, email, password):
        return self._post("signInWithPassword", {"email": email, "password": password, "returnSecureToken": True})

    def create_user_with_email_and_password(self, email, password):
        return self._post("signUp", {"email": email, "password": password, "returnSecureToken": True})

    def send_password_reset_email(self, email):
        return self._post("sendOobCode", {"requestType": "PASSWORD_RESET", "email": email})


class FirebaseApp:
    def __init__(self, api_key):
        self._auth = FirebaseAuth(api_key)

    def auth(self):
        return self._auth


@st.cache_resource(show_spinner=False)
def initialize_pyrebase():
    try:
        return FirebaseApp(st.secrets["FIREBASE"]["apiKey"])
    except Exception:
        logger.exception("Erro ao inicializar o Firebase Authentication")
        return None
