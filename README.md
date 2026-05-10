 🔍 LLM Hallucination Detector

Detects hallucinated claims in LLM responses by comparing them against a reference document using semantic similarity.

## Setup

```bash
pip install -r requirements.txt
```

## Run the pipeline test
```bash
python pipeline.py
```

## Run the Streamlit app
```bash
streamlit run app.py
```

## How it works
1. Splits LLM response into individual claims
2. Embeds claims + reference sentences using sentence-transformers
3. Finds the most semantically similar reference sentence for each claim
4. Labels each claim: Supported / Partial / Hallucinated based on cosine similarity score

## Thresholds
| Score | Label |
|---|---|
| >= 0.75 | ✅ Supported |
| 0.50 – 0.75 | ⚠️ Partial |
| < 0.50 | ❌ Hallucinated |

## Deploy to Hugging Face Spaces
1. Create a new Space (Streamlit SDK)
2. Upload `app.py`, `pipeline.py`, `requirements.txt`
3. Done — it runs automatically
