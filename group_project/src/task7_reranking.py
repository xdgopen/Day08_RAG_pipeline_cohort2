"""Task 7 — Reranking Module.

Improved offline reranker:
- Still lightweight/no external model.
- Boosts exact entity/keyphrase matches (e.g. "Miu Lê" -> "miu le").
- Penalizes candidates that do not contain the main named entity in the query.
- Adds intent bonuses for legal/crime questions such as "phạm tội gì".
"""

from __future__ import annotations

import re
from .task6_lexical_search import tokenize, normalize_text

STOPWORDS = {
    "toi", "muon", "hoi", "bi", "bat", "va", "pham", "toi", "gi", "la", "the", "nao",
    "cho", "biet", "ve", "cua", "co", "khong", "nhung", "cac", "theo",
}


def _query_entities(query: str) -> list[str]:
    """Extract simple Vietnamese name-like entities from a query.

    We keep this heuristic intentionally simple for offline mode. It catches names
    such as "Miu Lê", "Chi Dân", "An Tây", "Trúc Phương" and also quoted terms.
    """
    q_norm = normalize_text(query)
    known = [
        "miu le", "chi dan", "an tay", "truc phuong", "long nhat", "son ngoc minh",
        "son ngoc", "qwen", "ollama",
    ]
    found = [name for name in known if name in q_norm]

    # Capitalized phrase fallback from original query.
    for m in re.finditer(r"(?:[A-ZĐ][\wÀ-ỹ]+(?:\s+|$)){1,4}", query):
        phrase = normalize_text(m.group(0)).strip()
        toks = [t for t in phrase.split() if t not in STOPWORDS and len(t) > 1]
        if toks:
            candidate = " ".join(toks)
            if candidate and candidate not in found:
                found.append(candidate)
    return found


def _intent_bonus(query_norm: str, text_norm: str) -> float:
    bonus = 0.0
    if any(p in query_norm for p in ["pham toi", "toi gi", "hanh vi", "bi bat"]):
        for phrase, value in [
            ("su dung trai phep chat ma tuy", 0.35),
            ("to chuc su dung trai phep", 0.25),
            ("tang tru trai phep", 0.2),
            ("mua ban trai phep", 0.2),
            ("bi giu de dieu tra", 0.25),
            ("bat qua tang", 0.25),
            ("bi khoi to", 0.2),
            ("bi truy to", 0.2),
        ]:
            if phrase in text_norm:
                bonus += value
    return bonus


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    q_tokens = set(tokenize(query))
    q_norm = normalize_text(query)
    entities = _query_entities(query)
    out = []

    for c in candidates:
        content = c.get("content", "")
        text_norm = normalize_text(content)
        d_tokens = set(tokenize(content))
        overlap = len(q_tokens & d_tokens) / (len(q_tokens) or 1)
        base = float(c.get("score", 0.0))

        entity_bonus = 0.0
        entity_penalty = 0.0
        if entities:
            matches = sum(1 for ent in entities if ent in text_norm)
            if matches:
                entity_bonus = 0.55 + 0.15 * matches
            else:
                # If query clearly asks about a person, irrelevant person/crime chunks
                # should not outrank the target article.
                entity_penalty = 0.45

        title_source = normalize_text(str((c.get("metadata", {}) or {}).get("source", "")))
        source_bonus = 0.15 if any(ent.replace(" ", "-") in title_source or ent in title_source for ent in entities) else 0.0
        intent = _intent_bonus(q_norm, text_norm)

        md = c.get("metadata", {}) or {}
        doc_type = md.get("type", "")
        legal_bonus = 0.0
        news_penalty = 0.0
        if any(p in q_norm for p in ["dieu", "luat", "hinh phat", "quy dinh", "bo luat", "cai nghien"]):
            if doc_type == "legal":
                legal_bonus += 0.75
            elif doc_type == "news":
                news_penalty += 0.25
            if "dieu 249" in q_norm and ("dieu 249" in text_norm or "bo-luat-hinh-su" in title_source):
                legal_bonus += 0.65
            if "cai nghien" in q_norm and "cai nghien" in text_norm:
                legal_bonus += 0.45

        score = 0.45 * overlap + 0.20 * base + entity_bonus + source_bonus + intent + legal_bonus - entity_penalty - news_penalty
        item = c.copy()
        item["score"] = float(score)
        out.append(item)

    return sorted(out, key=lambda x: x.get("score", 0), reverse=True)[:top_k]


def _jaccard(a, b):
    sa = set(tokenize(a)); sb = set(tokenize(b))
    return len(sa & sb) / (len(sa | sb) or 1)


def rerank_mmr(query_embedding, candidates: list[dict], top_k: int = 5, lambda_param: float = 0.7) -> list[dict]:
    selected = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best = None
        best_score = -10**9
        for idx in remaining:
            rel = float(candidates[idx].get("score", 0.0))
            div = max((_jaccard(candidates[idx]["content"], candidates[j]["content"]) for j in selected), default=0.0)
            score = lambda_param * rel - (1 - lambda_param) * div
            if score > best_score:
                best_score = score
                best = idx
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    rrf_scores = {}
    content_map = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            key = item.get("content", "")
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map or item.get("score", 0) > content_map[key].get("score", 0):
                content_map[key] = item
    results = []
    for content, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder") -> list[dict]:
    if not candidates:
        return []
    if method == "rrf" and candidates and isinstance(candidates[0], list):
        return rerank_rrf(candidates, top_k)
    if method in ("cross_encoder", "mmr", "rrf"):
        return rerank_cross_encoder(query, candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    print(rerank("Miu Lê bị bắt và phạm tội gì?", [{"content": "Miu Lê bị giữ để điều tra hành vi sử dụng trái phép chất ma túy", "score": .2, "metadata": {}}]))
