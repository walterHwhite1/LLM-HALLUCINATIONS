import streamlit as st
from pipeline import detect_hallucinations

st.set_page_config(page_title="Hallucination Detector", page_icon="🔍", layout="wide")

st.title("🔍 LLM Hallucination Detector")
st.caption("Paste a reference document and an AI-generated response to see which claims are hallucinated.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Reference Document")
    reference = st.text_area("Paste the ground-truth document here", height=300,
        placeholder="e.g. a Wikipedia article, research paper, or any factual source...")

with col2:
    st.subheader("🤖 LLM Response")
    llm_response = st.text_area("Paste the AI-generated answer here", height=300,
        placeholder="e.g. a ChatGPT or Claude response about the document above...")

if st.button("Analyze", use_container_width=True, type="primary"):
    if not reference.strip() or not llm_response.strip():
        st.warning("Please fill in both fields.")
    else:
        with st.spinner("Analyzing claims..."):
            results, summary = detect_hallucinations(reference, llm_response)

        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Claims", summary["total_claims"])
        m2.metric("Supported", summary["supported"])
        m3.metric("Partial", summary["partial"])
        m4.metric("Hallucinated", summary["hallucinated"])

        rate = summary["hallucination_rate"]
        color = "normal" if rate < 20 else "inverse"
        st.progress(int(rate), text=f"Hallucination Rate: {rate}%")

        st.subheader("Claim-by-Claim Breakdown")

        label_colors = {
            "supported": ("✅", "green"),
            "partial": ("⚠️", "orange"),
            "hallucinated": ("❌", "red"),
        }

        for r in results:
            icon, color = label_colors[r["label"]]
            with st.expander(f"{icon} {r['claim'][:90]}{'...' if len(r['claim']) > 90 else ''} — score: {r['score']}"):
                st.markdown(f"**Full claim:** {r['claim']}")
                st.markdown(f"**Label:** :{color}[{r['label'].upper()}]")
                st.markdown(f"**Similarity score:** `{r['score']}`")
                st.markdown(f"**Closest match in document:**\n> {r['best_match']}")
