import os
import streamlit as st
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@st.cache_resource(show_spinner=True)
def initialize_system():
    try:
        # Configuração do modelo de embedding
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            normalize=True
        )
        Settings.llm = None  # Desliga o LLM interno da LlamaIndex (não será usado aqui)

        persist_dir = "db_chroma_enhanced"

        # Verifica se a pasta do índice já existe e tem conteúdo
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            st.warning("Gerando índice pela primeira vez... Isso pode levar alguns minutos.")

            # Carregar documentos .txt com metadados
            documents = SimpleDirectoryReader(
                input_dir="./arquivosFormatados",
                file_metadata=lambda x: {
                    "file_name": os.path.basename(x),
                    "file_path": x
                }
            ).load_data()

            # Splitter mais leve (baseado em sentenças)
            text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)

            # Quebrar documentos em chunks
            all_nodes = text_splitter.get_nodes_from_documents(documents)

            # Criar armazenamento com ChromaDB
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            chroma_collection = chroma_client.get_or_create_collection(name="enhanced_docs")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Criar e salvar índice
            index = VectorStoreIndex(all_nodes, storage_context=storage_context)
            index.storage_context.persist(persist_dir=persist_dir)

        # Carregar índice já existente
        index = VectorStoreIndex.from_persist_dir(persist_dir=persist_dir)
        return index.as_retriever(similarity_top_k=10)

    except Exception as e:
        st.error(f"Erro ao inicializar o sistema de RAG: {str(e)}")
        return None
