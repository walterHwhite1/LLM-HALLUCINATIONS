from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

SUPPORTED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.50

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def score_label(score):
    if score >= SUPPORTED_THRESHOLD:
        return "supported"
    elif score >= PARTIAL_THRESHOLD:
        return "partial"
    else:
        return "hallucinated"

def detect_hallucinations(reference: str, llm_response: str):
    ref_sentences = split_into_sentences(reference)
    claims = split_into_sentences(llm_response)

    ref_embeddings = model.encode(ref_sentences)
    claim_embeddings = model.encode(claims)

    results = []
    for i, claim in enumerate(claims):
        sims = cosine_similarity([claim_embeddings[i]], ref_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        label = score_label(best_score)
        results.append({
            "claim": claim,
            "label": label,
            "score": round(best_score, 3),
            "best_match": ref_sentences[best_idx],
        })

    total = len(results)
    hallucinated = sum(1 for r in results if r["label"] == "hallucinated")
    partial = sum(1 for r in results if r["label"] == "partial")
    supported = sum(1 for r in results if r["label"] == "supported")

    summary = {
        "total_claims": total,
        "supported": supported,
        "partial": partial,
        "hallucinated": hallucinated,
        "hallucination_rate": round((hallucinated / total) * 100, 1) if total else 0,
    }

    return results, summary


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

    results, summary = detect_hallucinations(reference, llm_response)

    print("\n=== CLAIM-BY-CLAIM ANALYSIS ===\n")
    icons = {"supported": "✅", "partial": "⚠️", "hallucinated": "❌"}
    for r in results:
        print(f"{icons[r['label']]} [{r['label'].upper()}] (score: {r['score']})")
        print(f"   Claim : {r['claim']}")
        print(f"   Closest: {r['best_match']}\n")

    print("=== SUMMARY ===")
    print(f"Total claims    : {summary['total_claims']}")
    print(f"✅ Supported    : {summary['supported']}")
    print(f"⚠️  Partial      : {summary['partial']}")
    print(f"❌ Hallucinated : {summary['hallucinated']}")
    print(f"Hallucination rate: {summary['hallucination_rate']}%")
