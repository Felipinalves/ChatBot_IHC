"""Identidade visual da UTFPR (Manual de Uso da Marca da UTFPR, 2024).

- Versão prioritária: logo colorida (preto + amarelo Pantone 7406 C) sobre fundo branco.
- Em tamanhos reduzidos, o texto "UNIVERSIDADE TECNOLÓGICA FEDERAL DO PARANÁ" deve
  ser retirado por falta de legibilidade: por isso o menu usa apenas o logotipo.
- Os SVGs em assets/ foram extraídos dos vetores do próprio manual, sem alterações
  de cor ou proporção.
"""
import base64
import os
from functools import lru_cache

import streamlit as st

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
LOGO_FULL = os.path.join(ASSETS_DIR, "utfpr_logo.svg")
LOGO_SYMBOL = os.path.join(ASSETS_DIR, "utfpr_simbolo.svg")

UTFPR_BLACK = "#000000"
UTFPR_YELLOW = "#FFCC00"  # CMYK 0/20/100/0 – Pantone 7406 C


@lru_cache(maxsize=None)
def _data_uri(path):
    with open(path, "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode()


def render_logo(width, css_class="", full=True):
    """Exibe a logo da UTFPR centralizada, preservando suas proporções."""
    src = _data_uri(LOGO_FULL if full else LOGO_SYMBOL)
    st.markdown(
        f"<div class='utfpr-logo {css_class}'>"
        f"<img src='{src}' width='{width}' alt='Logo da UTFPR – Universidade Tecnológica Federal do Paraná'>"
        f"</div>",
        unsafe_allow_html=True,
    )


def apply_brand_logo():
    """Logotipo da UTFPR no topo do menu lateral (e no cabeçalho com o menu recolhido)."""
    st.logo(LOGO_SYMBOL, size="large", link="https://www.utfpr.edu.br", icon_image=LOGO_SYMBOL)


def apply_styles():
    st.markdown(
        f"""
        <style>
        .utfpr-logo {{ display: flex; justify-content: center; padding: 24px 12px; background: #FFFFFF; }}
        .utfpr-logo img {{ max-width: 100%; height: auto; }}
        .utfpr-logo-login {{ margin-top: 1rem; }}
        .app-title {{ text-align: center; font-size: 2rem !important; padding: 0 !important; margin: 0; }}
        .app-subtitle {{ text-align: center; color: #555555; margin: .25rem 0 1.5rem; }}
        .app-footer {{ text-align: center; color: #777777; font-size: .8rem; margin-top: 2rem; }}
        [data-testid="stSidebar"] {{ border-right: 4px solid {UTFPR_YELLOW}; }}
        [data-testid="stSidebar"] .stButton button {{ justify-content: flex-start; text-align: left; }}
        .chat-current {{
            border-left: 4px solid {UTFPR_YELLOW}; background: #FFFFFF; border-radius: .5rem;
            padding: .5rem .75rem; font-weight: 600; font-size: .9rem; margin-bottom: .5rem;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }}
        [data-testid="stSidebarHeader"] img, [data-testid="stHeaderLogo"] {{ height: 2.5rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
