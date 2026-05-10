"""
Hallucination Detector Pipeline
--------------------------------
3-layer detection system:
  Layer 1 — Semantic similarity (sentence-transformers)
  Layer 2 — NLI entailment model (DeBERTa) — dedicated hallucination detection
  Layer 3 — Claude API verification (for borderline cases only)

Final verdict is decided by combining all three signals.
"""

import re
import os
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline as hf_pipeline

# ── Models (loaded once at startup) ────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# DeBERTa fine-tuned on NLI — classifies premise/hypothesis as:
#   ENTAILMENT   → claim is supported by reference
#   NEUTRAL      → claim is neither supported nor contradicted
#   CONTRADICTION → claim directly contradicts reference
nli_model = hf_pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-base",
    device=-1,  # CPU; change to 0 for GPU
)

# ── Thresholds ──────────────────────────────────────────────────────────────────
SIM_SUPPORTED  = 0.75
SIM_PARTIAL    = 0.50
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-sonnet-4-20250514"


# ── Helpers ─────────────────────────────────────────────────────────────────────
def split_into_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def sim_label(score: float) -> str:
    if score >= SIM_SUPPORTED:
        return "supported"
    elif score >= SIM_PARTIAL:
        return "partial"
    return "hallucinated"


# ── Layer 2: NLI (DeBERTa) ─────────────────────────────────────────────────────
def nli_check(claim: str, context: str) -> dict:
    result = nli_model(
        sequences=claim,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}",
        multi_label=False,
    )
    scores = dict(zip(result["labels"], result["scores"]))

    entail = scores.get("entailment", 0)
    neutral = scores.get("neutral", 0)
    contra  = scores.get("contradiction", 0)

    if entail > 0.5:
        label = "supported"
    elif contra > 0.4:
        label = "hallucinated"
    else:
        label = "partial"

    return {
        "nli_label": label,
        "entailment": round(entail, 3),
        "neutral": round(neutral, 3),
        "contradiction": round(contra, 3),
    }


# ── Layer 3: Claude API verification ───────────────────────────────────────────
def claude_verify(claim: str, context: str, api_key: str) -> dict:
    prompt = f"""You are a fact-checking assistant. Your job is to determine whether a CLAIM is supported by a REFERENCE TEXT.

REFERENCE TEXT:
{context}

CLAIM TO VERIFY:
{claim}

Respond ONLY in this exact JSON format, nothing else:
{{
  "verdict": "supported" | "partial" | "hallucinated",
  "explanation": "one sentence reason"
}}

Rules:
- "supported" = the reference clearly backs the claim
- "partial" = the reference vaguely relates but does not fully confirm
- "hallucinated" = the claim contradicts the reference or adds facts not in it"""

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 150,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        resp = requests.post(CLAUDE_API_URL, headers=headers, json=body, timeout=20)
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip()
        text = re.sub(r"```json|```", "", text).strip()
        parsed = json.loads(text)
        return {
            "claude_verdict": parsed.get("verdict", "partial"),
            "claude_explanation": parsed.get("explanation", ""),
        }
    except Exception as e:
        return {"claude_verdict": "partial", "claude_explanation": f"API error: {e}"}


# ── Final verdict combiner ──────────────────────────────────────────────────────
def combine_verdicts(sim_lbl, nli_lbl, claude_lbl):
    if claude_lbl:
        votes = [sim_lbl, nli_lbl, claude_lbl]
        return max(set(votes), key=votes.count)
    if sim_lbl == nli_lbl:
        return sim_lbl
    if "hallucinated" in (sim_lbl, nli_lbl):
        return "partial"
    return "partial"


# ── Main pipeline ───────────────────────────────────────────────────────────────
def detect_hallucinations(reference, llm_response, api_key="", use_claude=False):
    ref_sentences   = split_into_sentences(reference)
    claims          = split_into_sentences(llm_response)
    ref_embeddings  = embedder.encode(ref_sentences)
    claim_embeddings = embedder.encode(claims)

    results = []

    for i, claim in enumerate(claims):
        # Layer 1: similarity
        sims       = cosine_similarity([claim_embeddings[i]], ref_embeddings)[0]
        best_idx   = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_match = ref_sentences[best_idx]
        sim_lbl    = sim_label(best_score)

        # Layer 2: NLI
        nli_result = nli_check(claim, best_match)
        nli_lbl    = nli_result["nli_label"]

        # Layer 3: Claude (borderline only)
        is_borderline = sim_lbl == "partial" or (sim_lbl != nli_lbl)
        claude_result = {}
        claude_lbl    = None

        if use_claude and api_key and is_borderline:
            top3_idx       = np.argsort(sims)[-3:][::-1]
            context_window = " ".join(ref_sentences[j] for j in top3_idx)
            claude_result  = claude_verify(claim, context_window, api_key)
            claude_lbl     = claude_result["claude_verdict"]

        final = combine_verdicts(sim_lbl, nli_lbl, claude_lbl)

        results.append({
            "claim": claim,
            "label": final,
            "sim_score": round(best_score, 3),
            "sim_label": sim_lbl,
            "best_match": best_match,
            "nli_label": nli_lbl,
            "nli_entailment": nli_result["entailment"],
            "nli_contradiction": nli_result["contradiction"],
            "nli_neutral": nli_result["neutral"],
            "claude_used": bool(claude_lbl),
            "claude_verdict": claude_lbl or "—",
            "claude_explanation": claude_result.get("claude_explanation", "—"),
        })

    total       = len(results)
    hallucinated = sum(1 for r in results if r["label"] == "hallucinated")
    partial      = sum(1 for r in results if r["label"] == "partial")
    supported    = sum(1 for r in results if r["label"] == "supported")

    summary = {
        "total_claims": total,
        "supported": supported,
        "partial": partial,
        "hallucinated": hallucinated,
        "hallucination_rate": round((hallucinated / total) * 100, 1) if total else 0,
        "claude_calls_made": sum(1 for r in results if r["claude_used"]),
    }

    return results, summary


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reference = """
    The Eiffel Tower is located in Paris, France. It was built between 1887 and 1889
    as the entrance arch for the 1889 World's Fair. The tower is 330 metres tall and
    was designed by Gustave Eiffel. It attracts around 7 million visitors per year,
    making it the most visited paid monument in the world.
    """
    llm_response = """
    The Eiffel Tower is in Paris and stands 330 metres tall. It was constructed in 1850
    for the World's Fair. The tower was designed by Leonardo da Vinci and receives about
    7 million visitors annually. It is made entirely of copper.
    """

    API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
    USE_CLAUDE = bool(API_KEY)

    results, summary = detect_hallucinations(reference, llm_response, api_key=API_KEY, use_claude=USE_CLAUDE)

    icons = {"supported": "✅", "partial": "⚠️", "hallucinated": "❌"}
    print("\n=== CLAIM-BY-CLAIM ANALYSIS ===\n")
    for r in results:
        print(f"{icons[r['label']]} [{r['label'].upper()}]")
        print(f"   Claim      : {r['claim']}")
        print(f"   Sim        : {r['sim_score']} → {r['sim_label']}")
        print(f"   NLI        : entail={r['nli_entailment']} contra={r['nli_contradiction']} → {r['nli_label']}")
        if r["claude_used"]:
            print(f"   Claude     : {r['claude_verdict']} — {r['claude_explanation']}")
        print(f"   Best match : {r['best_match']}\n")

    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
