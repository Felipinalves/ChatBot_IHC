import streamlit as st

from auth.auth_utils import login_with_email, register_user, reset_password
from utils.branding import render_logo


def _start_session(user, email):
    st.session_state.user_id = user["localId"]
    st.session_state.user_email = email
    st.session_state.id_token = user["idToken"]


def show_auth_page(auth, firestore_db):
    """Exibe a tela inicial de autenticação."""
    _, center, _ = st.columns([1, 2, 1])
    with center:
        render_logo(width=340, css_class="utfpr-logo-login")
        st.markdown(
            "<h1 class='app-title'>IAHC ChatBot</h1>"
            "<p class='app-subtitle'>Assistente de estudos em Interação Humano-Computador</p>",
            unsafe_allow_html=True,
        )

        if auth is None or firestore_db is None:
            st.error("O serviço de autenticação está indisponível no momento. Tente novamente mais tarde.")
            return

        login_tab, register_tab, reset_tab = st.tabs(["Entrar", "Criar conta", "Recuperar senha"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("E-mail", key="login_email")
                password = st.text_input("Senha", type="password", key="login_password")
                if st.form_submit_button("Entrar", type="primary", use_container_width=True):
                    if email and password:
                        with st.spinner("Entrando..."):
                            user, error = login_with_email(auth, email.strip(), password)
                        if user:
                            _start_session(user, email.strip())
                            st.rerun()
                        else:
                            st.error(error)
                    else:
                        st.warning("Por favor, preencha todos os campos.")

        with register_tab:
            with st.form("register_form"):
                name = st.text_input("Nome", key="register_name")
                email = st.text_input("E-mail", key="register_email")
                password = st.text_input(
                    "Senha", type="password", key="register_password", help="Mínimo de 6 caracteres."
                )
                if st.form_submit_button("Criar conta", type="primary", use_container_width=True):
                    if name and email and password:
                        with st.spinner("Criando conta..."):
                            user, error = register_user(
                                auth, firestore_db, email.strip(), password, name.strip()
                            )
                        if user:
                            _start_session(user, email.strip())
                            st.rerun()
                        else:
                            st.error(error)
                    else:
                        st.warning("Por favor, preencha todos os campos.")

        with reset_tab:
            with st.form("reset_form"):
                email = st.text_input("E-mail", key="reset_email")
                if st.form_submit_button("Enviar link de recuperação", use_container_width=True):
                    if email:
                        with st.spinner("Enviando e-mail de recuperação..."):
                            success, message = reset_password(auth, email.strip())
                        (st.success if success else st.error)(message)
                    else:
                        st.warning("Por favor, informe seu e-mail.")

        st.markdown(
            "<p class='app-footer'>Universidade Tecnológica Federal do Paraná</p>",
            unsafe_allow_html=True,
        )
