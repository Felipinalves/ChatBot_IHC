import os
from llama_index.core import SimpleDirectoryReader, SentenceSplitter, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

@st.cache_resource(show_spinner=False)
def initialize_system():
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            normalize=True
        )
        Settings.llm = None

        persist_dir = "db_chroma_enhanced"

        # Se o diretório persistido não existe, construir o índice
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            st.warning("Gerando índice inicial... Isso pode demorar alguns minutos.")
            
            documents = SimpleDirectoryReader(
                input_dir="./arquivosFormatados",
                file_metadata=lambda x: {
                    "file_name": os.path.basename(x),
                    "file_path": x
                }
            ).load_data()

            text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
            all_nodes = text_splitter.get_nodes_from_documents(documents)

            chroma_client = chromadb.PersistentClient(path=persist_dir)
            chroma_collection = chroma_client.get_or_create_collection(name="enhanced_docs")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(all_nodes, storage_context=storage_context)
            index.storage_context.persist(persist_dir=persist_dir)

        # Carregar índice salvo
        index = VectorStoreIndex.from_persist_dir(persist_dir=persist_dir)
        return index.as_retriever(similarity_top_k=10)

    except Exception as e:
        st.error(f"Erro ao carregar índice: {str(e)}")
        return None
