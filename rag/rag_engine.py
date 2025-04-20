import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

@st.cache_resource(show_spinner=False)
def initialize_system():
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            normalize=True
        )
        Settings.llm = None

        index = VectorStoreIndex.from_persist_dir(persist_dir="db_chroma_enhanced")
        return index.as_retriever(similarity_top_k=10)

    except Exception as e:
        st.error(f"Erro ao carregar índice: {str(e)}")
        return None
