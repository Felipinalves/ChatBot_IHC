"""Gera o índice vetorial pré-calculado a partir de ./arquivosFormatados.

Rode localmente sempre que o acervo de artigos mudar e faça commit da pasta
rag/index/. O app em produção apenas carrega esses arquivos, sem recalcular
embeddings a cada inicialização (o que consumia muita CPU no Streamlit Cloud).

Uso:
    pip install -r requirements-dev.txt
    python -m rag.build_index
"""
import json
import os
import time

import numpy as np

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    DOCS_DIR,
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    INDEX_DIR,
    META_FILE,
)


def load_documents(docs_dir=DOCS_DIR):
    """Lê todos os .txt do acervo, em ordem alfabética para o índice ser reprodutível."""
    documents = []
    for name in sorted(os.listdir(docs_dir)):
        if not name.lower().endswith(".txt"):
            continue
        with open(os.path.join(docs_dir, name), encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if text:
            documents.append((name, text))
    return documents


def split_documents(documents):
    """Divide os documentos em trechos usando o mesmo splitter da versão anterior."""
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for file_name, text in documents:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append({"file_name": file_name, "text": piece})
    return chunks


def main():
    start = time.time()
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"{len(documents)} documentos -> {len(chunks)} trechos")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float16)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "documents": len(documents),
                "chunks": len(chunks),
                "dimensions": int(embeddings.shape[1]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Índice salvo em {INDEX_DIR} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
