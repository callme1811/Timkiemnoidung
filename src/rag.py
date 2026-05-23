import re

import google.generativeai as genai
import numpy as np

from .config import EMBEDDING_MODEL


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_score(question: str, text: str, title: str = "") -> int:
    q_words = re.findall(r"\w+", normalize_text(question))
    content = normalize_text(f"{title} {text}")

    if not q_words:
        return 0

    score = 0
    for word in q_words:
        if word in content:
            score += 2

    question_norm = normalize_text(question)
    if question_norm and question_norm in content:
        score += 5

    return score


def select_relevant_nodes(question: str, nodes: list[dict], top_k: int) -> list[dict]:
    if not nodes:
        return []

    ranked = sorted(
        nodes,
        key=lambda n: keyword_score(question, n.get("text", ""), n.get("title", "")),
        reverse=True,
    )

    selected = [
        n for n in ranked
        if keyword_score(question, n.get("text", ""), n.get("title", "")) > 0
    ]

    if not selected:
        selected = ranked

    return selected[:top_k]


def cosine_similarity(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0
    return float(np.dot(a, b) / norm)


def get_embeddings_for_nodes(nodes: list[dict], api_key: str, embedding_cache: dict | None = None):
    genai.configure(api_key=api_key)
    embedding_cache = embedding_cache if embedding_cache is not None else {}
    uncached_nodes = [n for n in nodes if n["node_id"] not in embedding_cache]

    if uncached_nodes:
        batch_size = 50
        for i in range(0, len(uncached_nodes), batch_size):
            batch = uncached_nodes[i:i + batch_size]
            texts = [n["text"] for n in batch]

            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts,
                task_type="retrieval_document",
            )

            for idx, node in enumerate(batch):
                embedding_cache[node["node_id"]] = result["embedding"][idx]

    return embedding_cache


def select_relevant_nodes_semantic(
    question: str,
    nodes: list[dict],
    top_k: int,
    api_key: str,
    embedding_cache: dict | None = None,
):
    if not nodes:
        return [], embedding_cache or {}

    embedding_cache = get_embeddings_for_nodes(nodes, api_key, embedding_cache)

    genai.configure(api_key=api_key)
    q_result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=question,
        task_type="retrieval_query",
    )
    q_emb = q_result["embedding"]

    scored_nodes = []
    for node in nodes:
        node_id = node["node_id"]
        if node_id in embedding_cache:
            score = cosine_similarity(q_emb, embedding_cache[node_id])
            scored_nodes.append((node, score))

    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored_nodes[:top_k]], embedding_cache


def build_context(selected_nodes: list[dict]) -> str:
    return "\n\n".join([n.get("text", "") for n in selected_nodes])
