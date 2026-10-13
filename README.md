# IAHC ChatBot – UTFPR

Assistente de estudos em Interação Humano-Computador (IHC). Responde perguntas em português
com base em um acervo de artigos científicos da área, usando RAG (*Retrieval-Augmented
Generation*) e o modelo Google Gemini.

App publicado: https://iahc-chatbot.streamlit.app/

## Como funciona

1. Os artigos em `arquivosFormatados/` são divididos em trechos e convertidos em vetores com o
   modelo `paraphrase-multilingual-mpnet-base-v2`. Esse índice é **gerado uma vez** e versionado
   em `rag/index/`, para o app não recalcular tudo a cada inicialização.
2. A cada pergunta, o app busca os trechos mais parecidos (similaridade de cosseno) e envia ao
   Gemini junto com o histórico recente da conversa.
3. Login, cadastro e histórico de conversas usam Firebase Authentication e Firestore.
   Conversas sem atividade por 30 dias são excluídas automaticamente.

Para perguntar sobre um artigo específico: `artigo: "nome do artigo"`.

## Secrets (Streamlit Cloud → Settings → Secrets, ou `.streamlit/secrets.toml`)

```toml
GEMINI_API_KEY = "..."           # chave do Google AI Studio
GEMINI_MODEL = "gemini-3.1-flash-lite"  # opcional

[FIREBASE]
apiKey = "..."
authDomain = "..."
projectId = "..."
storageBucket = "..."
messagingSenderId = "..."
appId = "..."

[FIREBASE_SERVICE_ACCOUNT]
# conteúdo do JSON da conta de serviço do Firebase
```

## Rodando localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

## Atualizando o acervo de artigos

Depois de adicionar ou alterar arquivos em `arquivosFormatados/`, gere o índice novamente e faça
commit da pasta `rag/index/`:

```bash
python -m rag.build_index
```

## Testes

```bash
python -m pytest
```

Os testes usam Firebase e Gemini falsos, então não precisam de credenciais.

## Identidade visual

A logo da UTFPR em `assets/` foi extraída dos vetores do
[Manual de Uso da Marca da UTFPR](https://www.utfpr.edu.br/comunicacao/design/manual-de-uso-da-identidade-visual-da-utfpr):
versão prioritária (colorida, sobre fundo branco) na tela inicial e, no menu, o logotipo sem o
texto "Universidade Tecnológica Federal do Paraná", como o manual orienta para tamanhos reduzidos.
