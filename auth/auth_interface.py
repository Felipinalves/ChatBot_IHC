import streamlit as st
from auth.auth_utils import login_with_email, register_user, reset_password
from utils.time_utils import get_brasilia_time

def show_auth_page(auth, firestore_db):
    """Exibe a interface de autenticação"""
    
    st.title("🤖 IAHC ChatBot")
    
    # Tab para opções de login ou registro
    tab = st.tabs(["Login", "Cadastro", "Recuperar Senha"])
    
    with tab[0]:
        with st.form("login_form"):
            email = st.text_input("E-mail", key="login_email")
            password = st.text_input("Senha", type="password", key="login_password")
            submit_login = st.form_submit_button("Entrar")
            
            if submit_login:
                if email and password:
                    with st.spinner("Fazendo login..."):
                        user, error = login_with_email(auth, email, password)
                        
                        if user:
                            # Guardar informações importantes na sessão
                            st.session_state.user_id = user['localId']
                            st.session_state.user_email = email  # Armazenar o email para uso no dropdown
                            st.session_state.id_token = user['idToken']
                            st.success("Login realizado com sucesso!")
                            st.rerun()
                        else:
                            st.error(error)
                else:
                    st.warning("Por favor, preencha todos os campos.")
    
    with tab[1]:
        with st.form("register_form"):
            name = st.text_input("Nome", key="register_name")
            email = st.text_input("E-mail", key="register_email")
            password = st.text_input("Senha", type="password", key="register_password")
            submit_register = st.form_submit_button("Cadastrar")
            
            if submit_register:
                if name and email and password:
                    with st.spinner("Criando conta..."):
                        user, error = register_user(auth, firestore_db, email, password, name)
                        
                        if user:
                            # Guardar informações importantes na sessão
                            st.session_state.user_id = user['localId']
                            st.session_state.user_email = email  # Armazenar o email para uso no dropdown
                            st.session_state.id_token = user['idToken']
                            st.success("Cadastro realizado com sucesso!")
                            st.rerun()
                        else:
                            st.error(error)
                else:
                    st.warning("Por favor, preencha todos os campos.")
    
    with tab[2]:
        with st.form("reset_form"):
            email = st.text_input("E-mail", key="reset_email")
            submit_reset = st.form_submit_button("Recuperar Senha")
            
            if submit_reset:
                if email:
                    with st.spinner("Enviando e-mail de recuperação..."):
                        success, message = reset_password(auth, email)
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.warning("Por favor, informe seu e-mail.")
