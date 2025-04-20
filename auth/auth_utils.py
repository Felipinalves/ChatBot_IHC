import json
import streamlit as st
from utils.time_utils import get_brasilia_time

def login_with_email(auth, email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return user, None
    except Exception as e:
        error_json = e.args[1]
        error_dict = json.loads(error_json)
        error_message = error_dict.get('error', {}).get('message', 'Erro desconhecido')
        
        if error_message == 'EMAIL_NOT_FOUND':
            return None, "E-mail não encontrado. Por favor, verifique seu e-mail ou crie uma conta."
        elif error_message == 'INVALID_LOGIN_CREDENTIALS':
            return None, "Credenciais erradas. Por favor, tente novamente."
        else:
            return None, f"Erro ao fazer login: {error_message}"

def register_user(auth, db, email, password, name):
    try:
        # Usando Pyrebase para autenticação
        user = auth.create_user_with_email_and_password(email, password)
        
        # Criando dados do usuário para o Firestore
        user_data = {
            "name": name,
            "email": email,
            "created_at": get_brasilia_time(),
            "photo_url": ""
        }
        
        # Usando Firestore para armazenar dados do usuário
        db.collection("users").document(user['localId']).set(user_data)
        
        return user, None
    except Exception as e:
        error_json = e.args[1]
        error_dict = json.loads(error_json)
        error_message = error_dict.get('error', {}).get('message', 'Erro desconhecido')
        
        if error_message == 'EMAIL_EXISTS':
            return None, f"Você já possui uma conta. Por favor, faça login ou recupere sua senha."
        elif 'WEAK_PASSWORD' in error_message:
            return None, f"Você inseriu uma senha fraca. Por favor insira uma senha com no mínimo 6 caracteres"
        else:
            return None, f"Erro ao criar conta: {error_message}"

def reset_password(auth, email):
    try:
        auth.send_password_reset_email(email)
        return True, "E-mail de recuperação enviado. Verifique sua caixa de entrada."
    except Exception as e:
        error_json = e.args[1]
        error_dict = json.loads(error_json)
        error_message = error_dict.get('error', {}).get('message', 'Erro desconhecido')
        
        if error_message == 'EMAIL_NOT_FOUND':
            return False, "E-mail não encontrado. Por favor, verifique seu e-mail ou crie uma conta."
        else:
            return False, f"Erro ao enviar e-mail de recuperação: {error_message}"

