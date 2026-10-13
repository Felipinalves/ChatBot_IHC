import json
import os
from dataclasses import dataclass

import numpy as np
import streamlit as st

from rag.config import (
    CHUNKS_FILE,
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    META_FILE,
    MIN_SCORE,
    TOP_K,
)


@dataclass
class RetrievedChunk:
    text: str
    file_name: str
    score: float


class Retriever:
    """Busca semântica sobre o índice pré-calculado (rag/index)."""

    def __init__(self, embeddings, chunks, encode):
        if len(embeddings) != len(chunks):
            raise ValueError("Índice inconsistente: número de embeddings e de trechos diferente.")
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.chunks = chunks
        self._encode = encode
        self.file_names = sorted({c["file_name"] for c in chunks})

    def find_file(self, name):
        """Encontra um artigo do acervo pelo nome (ignorando maiúsculas e a extensão)."""
        wanted = name.lower().removesuffix(".txt").strip()
        for file_name in self.file_names:
            if file_name.lower().removesuffix(".txt") == wanted:
                return file_name
        for file_name in self.file_names:
            if wanted and wanted in file_name.lower():
                return file_name
        return None

    def retrieve(self, query, top_k=TOP_K, min_score=MIN_SCORE, file_name=None):
        query_vec = np.asarray(self._encode(query), dtype=np.float32)
        scores = self.embeddings @ query_vec

        if file_name:
            mask = np.array([c["file_name"] == file_name for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)

        top = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(
                text=self.chunks[i]["text"],
                file_name=self.chunks[i]["file_name"],
                score=float(scores[i]),
            )
            for i in top
            if scores[i] >= min_score
        ]


def load_index():
    if not (os.path.exists(EMBEDDINGS_FILE) and os.path.exists(CHUNKS_FILE)):
        raise FileNotFoundError(
            "Índice não encontrado em rag/index. Gere-o com: python -m rag.build_index"
        )
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    if meta["embedding_model"] != EMBEDDING_MODEL:
        raise ValueError(
            "O índice foi gerado com outro modelo de embeddings. Gere-o novamente com: "
            "python -m rag.build_index"
        )
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    return np.load(EMBEDDINGS_FILE), chunks


@st.cache_resource(show_spinner="Carregando a base de conhecimento...")
def initialize_system():
    """Carrega o índice e o modelo de embeddings uma única vez por servidor."""
    import torch
    from sentence_transformers import SentenceTransformer

    # Poucas threads: o Streamlit Community Cloud oferece CPU limitada
    torch.set_num_threads(2)
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    def encode(text):
        return model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

    embeddings, chunks = load_index()
    return Retriever(embeddings, chunks, encode)
