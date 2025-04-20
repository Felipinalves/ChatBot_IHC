import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.chroma.base import ChromaVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st

# Download dos recursos do NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

@st.cache_resource(show_spinner=False)
def initialize_system():
    try:
        # Configuração avançada de embeddings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            normalize=True
        )
        Settings.llm = None

        # Inicializar o splitter semântico
        text_splitter = SemanticSplitterNodeParser(
            buffer_size=2,
            breakpoint_percentile_threshold=90,
            embed_model=Settings.embed_model
        )

        # Configuração otimizada do ChromaDB
        db = chromadb.PersistentClient(path="db_chroma_enhanced")
        chroma_collection = db.get_or_create_collection(
            name="enhanced_docs",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 48,
                "hnsw:construction_ef": 300,
                "hnsw:search_ef": 500
            }
        )

        # Carregar e processar documentos
        documents = SimpleDirectoryReader(
            "./arquivosFormatados",
            file_metadata=lambda x: _extract_metadata(x)
        ).load_data()

        # Processar documentos com splitter semântico
        processed_nodes = []
        for doc in documents:
            nodes = text_splitter.get_nodes_from_documents([doc])
            processed_nodes.extend(nodes)

        # Criar índice
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex(
            processed_nodes,
            storage_context=storage_context,
            show_progress=True
        )

        return index.as_retriever(similarity_top_k=20)
        
    except Exception as e:
        st.error(f"Erro na inicialização do sistema: {str(e)}")
        return None

def _extract_metadata(filename: str) -> dict:
    """Função básica de extração de metadados (personalizável)"""
    return {
        "file_name": filename.split("/")[-1],
        "file_type": filename.split(".")[-1],
        "file_path": filename
    }
