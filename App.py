import os
import streamlit as s
from dotenv import load_dotenv
from pipeline import detect_hallucinations
 
load_dotenv()
 
st.set_page_config(
    page_title="LLM Hallucination Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Hallucination Detector")
    st.caption("3-layer AI fact-checking system")
    st.divider()
 
    api_key = st.text_input(
        "Anthropic API Key (optional)",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Enables Claude verification. Get one at console.anthropic.com",
    )
    use_claude = st.toggle(
        "Enable Claude verification",
        value=bool(api_key),
    )
    st.divider()
    st.subheader("🧠 How it works")
    st.markdown("""
**Layer 1 — Semantic Similarity**
Finds the closest sentence in your reference using cosine similarity.
 
**Layer 2 — DeBERTa NLI**
Classifies each claim as entailment, neutral, or contradiction.
 
**Layer 3 — Claude API**
Only triggers on borderline cases where Layer 1 and 2 disagree.
 
**Final verdict** = majority vote across active layers.
""")
 
# ── Main UI ──────────────────────────────────────────────────────────────────────
st.title("LLM Hallucination Detector")
st.markdown("Paste your **own** reference document and an AI-generated response to fact-check it.")
st.divider()
 
with st.expander("📖 How to use", expanded=False):
    st.markdown("""
1. Paste any document in the **left box** (Wikipedia article, news, report — anything)
2. Ask ChatGPT/Claude/Gemini about that document and paste its response in the **right box**
3. Click **Analyze** — every sentence gets checked claim by claim
4. See which sentences are ✅ supported, ⚠️ uncertain, or ❌ hallucinated
""")
 
# ── Input boxes — NO session state, reads directly from user ──────────────────
col1, col2 = st.columns(2, gap="medium")
 
with col1:
    st.subheader("📄 Your Reference Document")
    st.caption("Paste the ground-truth source you want to fact-check against")
    reference = st.text_area(
        "ref",
        height=320,
        placeholder="Paste any article, Wikipedia page, research paper, report...",
        label_visibility="collapsed",
    )
 
with col2:
    st.subheader("🤖 LLM Response to Verify")
    st.caption("Paste the AI-generated answer you want to fact-check")
    llm_response = st.text_area(
        "llm",
        height=320,
        placeholder="Paste a ChatGPT, Claude, or Gemini response about the document on the left...",
        label_visibility="collapsed",
    )
 
if reference:
    st.caption(f"Reference: {len(reference.split())} words")
if llm_response:
    st.caption(f"LLM response: {len(llm_response.split())} words")
 
analyze = st.button("🔎 Analyze for Hallucinations", use_container_width=True, type="primary")
 
# ── Analysis — uses ONLY what is currently in the text boxes ─────────────────
if analyze:
    ref_text  = reference.strip()
    resp_text = llm_response.strip()
 
    if not ref_text:
        st.error("Please paste a reference document in the left box.")
        st.stop()
    if not resp_text:
        st.error("Please paste an LLM response in the right box.")
        st.stop()
    if len(ref_text.split()) < 10:
        st.warning("Reference is very short — paste at least a paragraph for accurate results.")
        st.stop()
    if len(resp_text.split()) < 5:
        st.warning("LLM response is very short — paste the full response.")
        st.stop()
 
    try:
        with st.spinner("Analyzing... (first run takes 30–60 seconds to load models)"):
            results, summary = detect_hallucinations(
                ref_text,
                resp_text,
                api_key=api_key if use_claude else "",
                use_claude=use_claude and bool(api_key),
            )
 
        # remove duplicate claims — same label + same claim text
        seen = set()
        unique_results = []
        for r in results:
            key = r["claim"].strip().lower()
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        results = unique_results
 
        # recalculate summary from deduplicated results
        total        = len(results)
        hallucinated = sum(1 for r in results if r["label"] == "hallucinated")
        partial      = sum(1 for r in results if r["label"] == "partial")
        supported    = sum(1 for r in results if r["label"] == "supported")
        summary = {
            "total_claims":       total,
            "supported":          supported,
            "partial":            partial,
            "hallucinated":       hallucinated,
            "hallucination_rate": round((hallucinated / total) * 100, 1) if total else 0,
            "claude_calls_made":  sum(1 for r in results if r["claude_used"]),
        }
 
        st.divider()
        st.subheader("📊 Results")
 
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Claims",    summary["total_claims"])
        m2.metric("✅ Supported",    summary["supported"])
        m3.metric("⚠️ Partial",      summary["partial"])
        m4.metric("❌ Hallucinated", summary["hallucinated"])
        m5.metric("Claude calls",    summary["claude_calls_made"])
 
        rate      = summary["hallucination_rate"]
        indicator = "🟢" if rate < 20 else "🟠" if rate < 50 else "🔴"
        risk_msg  = (
            "Low risk — response looks reliable" if rate < 20
            else "Moderate risk — verify key claims" if rate < 50
            else "High risk — do not trust without verification"
        )
        st.progress(min(int(rate), 100), text=f"{indicator} Hallucination Rate: **{rate}%** — {risk_msg}")
 
        st.divider()
 
        tab_all, tab_hall, tab_partial, tab_good = st.tabs([
            f"All ({total})",
            f"❌ Hallucinated ({hallucinated})",
            f"⚠️ Partial ({partial})",
            f"✅ Supported ({supported})",
        ])
 
        label_styles = {
            "supported":    ("✅", "green"),
            "partial":      ("⚠️", "orange"),
            "hallucinated": ("❌", "red"),
        }
 
        def render_claims(claims_list, empty_msg="Nothing to show here."):
            if not claims_list:
                st.info(empty_msg)
                return
            for r in claims_list:
                icon, color = label_styles[r["label"]]
                preview = r["claim"][:90] + ("..." if len(r["claim"]) > 90 else "")
                with st.expander(f"{icon} {preview}"):
                    st.markdown(f"**Full claim:** {r['claim']}")
                    st.markdown(f"**Verdict:** :{color}[**{r['label'].upper()}**]")
                    st.divider()
 
                    l1, l2 = st.columns(2)
                    with l1:
                        st.markdown("**Layer 1 — Similarity**")
                        st.code(f"Score : {r['sim_score']}\nLabel : {r['sim_label']}", language="")
                    with l2:
                        st.markdown("**Layer 2 — DeBERTa NLI**")
                        st.code(
                            f"Entailment   : {r['nli_entailment']}\n"
                            f"Neutral      : {r['nli_neutral']}\n"
                            f"Contradiction: {r['nli_contradiction']}\n"
                            f"Label        : {r['nli_label']}",
                            language="",
                        )
 
                    if r["claude_used"]:
                        st.markdown("**Layer 3 — Claude Verification**")
                        st.info(
                            f"**Verdict:** {r['claude_verdict'].upper()}  \n"
                            f"**Reason:** {r['claude_explanation']}"
                        )
                    else:
                        st.caption("Layer 3 — Claude not triggered (add API key in sidebar to enable)")
 
                    st.markdown(f"**Closest match in your document:**\n> _{r['best_match']}_")
 
        with tab_all:
            render_claims(results)
        with tab_hall:
            render_claims([r for r in results if r["label"] == "hallucinated"], "✅ No hallucinations detected!")
        with tab_partial:
            render_claims([r for r in results if r["label"] == "partial"], "No partial claims.")
        with tab_good:
            render_claims([r for r in results if r["label"] == "supported"], "⚠️ No fully supported claims found.")
 
    except ValueError as e:
        st.error(f"Input error: {e}")
    except RuntimeError as e:
        st.error(f"Model error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
 
