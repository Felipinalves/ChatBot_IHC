import json
import logging

from utils.time_utils import get_brasilia_time

logger = logging.getLogger(__name__)

FIREBASE_ERRORS = {
    "EMAIL_NOT_FOUND": "E-mail não encontrado. Verifique seu e-mail ou crie uma conta.",
    "INVALID_LOGIN_CREDENTIALS": "E-mail ou senha incorretos. Tente novamente.",
    "INVALID_PASSWORD": "E-mail ou senha incorretos. Tente novamente.",
    "INVALID_EMAIL": "E-mail inválido. Verifique o endereço digitado.",
    "MISSING_PASSWORD": "Informe a senha.",
    "USER_DISABLED": "Esta conta foi desativada.",
    "EMAIL_EXISTS": "Você já possui uma conta. Faça login ou recupere sua senha.",
    "WEAK_PASSWORD": "Senha fraca. Use uma senha com no mínimo 6 caracteres.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "Muitas tentativas. Aguarde alguns minutos e tente novamente.",
}


def firebase_error_message(exc, default):
    """Extrai o código de erro do Firebase e devolve uma mensagem amigável."""
    code = getattr(exc, "code", None)
    if code is None:
        try:
            code = json.loads(exc.args[1])["error"]["message"]
        except Exception:
            logger.exception("Erro inesperado do Firebase")
    if code:
        for key, message in FIREBASE_ERRORS.items():
            if code == key or code.startswith(key + " "):
                return message
        logger.warning("Erro do Firebase não mapeado: %s", code)
    return default


def login_with_email(auth, email, password):
    try:
        return auth.sign_in_with_email_and_password(email, password), None
    except Exception as e:
        return None, firebase_error_message(e, "Não foi possível fazer login. Tente novamente.")


def register_user(auth, db, email, password, name):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        db.collection("users").document(user["localId"]).set(
            {
                "name": name,
                "email": email,
                "created_at": get_brasilia_time(),
                "photo_url": "",
            }
        )
        return user, None
    except Exception as e:
        return None, firebase_error_message(e, "Não foi possível criar a conta. Tente novamente.")


def reset_password(auth, email):
    try:
        auth.send_password_reset_email(email)
        return True, "E-mail de recuperação enviado. Verifique sua caixa de entrada."
    except Exception as e:
        return False, firebase_error_message(
            e, "Não foi possível enviar o e-mail de recuperação. Tente novamente."
        )
