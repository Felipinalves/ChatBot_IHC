import logging
import re
import time

import streamlit as st

logger = logging.getLogger(__name__)

# gemini-2.5-flash-lite está sendo descontinuado; o modelo pode ser trocado
# sem alterar o código definindo GEMINI_MODEL nos secrets do Streamlit.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_output_tokens": 4096,
}

_API_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{30,}")


class GeminiError(Exception):
    """Erro com uma mensagem segura para exibir ao usuário."""

    def __init__(self, user_message, retryable=False):
        super().__init__(user_message)
        self.user_message = user_message
        self.retryable = retryable


def redact(text):
    """Remove chaves de API de mensagens que vão para logs ou para a tela."""
    return _API_KEY_PATTERN.sub("[CHAVE OCULTA]", str(text))


def to_user_error(exc):
    """Traduz erros da API do Gemini em mensagens claras, sem expor a chave."""
    code = getattr(exc, "code", None)
    raw = redact(exc)
    logger.error("Erro na API do Gemini (%s): %s", code, raw)

    if code in (401, 403) or "PERMISSION_DENIED" in raw or "API_KEY_INVALID" in raw:
        return GeminiError(
            "O serviço de IA está indisponível: a chave da API do Gemini é inválida, foi "
            "suspensa ou não tem permissão. O administrador precisa gerar uma nova chave "
            "no Google AI Studio e atualizar os secrets do app."
        )
    if code == 429 or "RESOURCE_EXHAUSTED" in raw:
        return GeminiError(
            "O limite de uso do serviço de IA foi atingido. Aguarde alguns instantes e tente novamente.",
            retryable=True,
        )
    if code == 404 or "NOT_FOUND" in raw:
        return GeminiError(
            "O modelo de IA configurado não está disponível. O administrador deve atualizar "
            "GEMINI_MODEL nos secrets do app."
        )
    if code is not None and code >= 500:
        return GeminiError(
            "O serviço de IA está instável no momento. Tente novamente em instantes.",
            retryable=True,
        )
    return GeminiError("Não foi possível gerar a resposta. Tente novamente em instantes.")


@st.cache_resource(show_spinner=False)
def _get_client(api_key):
    from google import genai

    return genai.Client(api_key=api_key)


def _settings():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError(
            "O serviço de IA não está configurado: defina GEMINI_API_KEY nos secrets do app."
        )
    return api_key, st.secrets.get("GEMINI_MODEL", DEFAULT_MODEL)


def stream_response(contents, system_instruction=None, max_retries=3):
    """Gera a resposta do Gemini em partes (streaming).

    Erros transitórios são repetidos com espera crescente, mas só antes de o
    primeiro trecho ser enviado, para não duplicar texto na tela.
    """
    from google.genai import types

    api_key, model = _settings()
    client = _get_client(api_key)
    config = types.GenerateContentConfig(system_instruction=system_instruction, **GENERATION_CONFIG)

    for attempt in range(max_retries):
        started = False
        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=contents, config=config
            ):
                if chunk.text:
                    started = True
                    yield chunk.text
            return
        except GeminiError:
            raise
        except Exception as exc:
            error = to_user_error(exc)
            if started or not error.retryable or attempt == max_retries - 1:
                raise error from None
            time.sleep(2 ** attempt)
