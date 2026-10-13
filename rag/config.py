import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCS_DIR = os.path.join(ROOT_DIR, "arquivosFormatados")
INDEX_DIR = os.path.join(ROOT_DIR, "rag", "index")
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, "embeddings.npy")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")
META_FILE = os.path.join(INDEX_DIR, "meta.json")

# O mesmo modelo precisa ser usado para gerar o índice e para as perguntas
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# Recuperação
TOP_K = 7
MIN_SCORE = 0.45  # com 0.6, trechos claramente relevantes (~0.55) eram descartados
