# Roda uma vez para criar e salvar o índice com os embeddings
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# Configura embedding (multilingue e leve)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    normalize=True
)
Settings.llm = None

# Carregar documentos
documents = SimpleDirectoryReader(
    input_dir="./arquivosFormatados",
    file_metadata=lambda x: {
        "file_name": os.path.basename(x),
        "file_path": x
    }
).load_data()

# Splitter mais leve e rápido
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)

# Processar
all_nodes = text_splitter.get_nodes_from_documents(documents)

# ChromaDB
chroma_client = chromadb.PersistentClient(path="db_chroma_enhanced")
chroma_collection = chroma_client.get_or_create_collection(name="enhanced_docs")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Construir e salvar o índice
index = VectorStoreIndex(all_nodes, storage_context=storage_context)
index.storage_context.persist(persist_dir="db_chroma_enhanced")
print("\u2705 Índice salvo com sucesso!")
