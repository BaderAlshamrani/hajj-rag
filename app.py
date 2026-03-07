"""
Hajj Safety Guide — RAG Streamlit App
"""

import json
import time
import requests
import numpy as np
import streamlit as st
import google.genai as genai
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# API keys come from st.secrets (Streamlit Cloud) or fallback to hardcoded for local dev
FIREWORKS_API_KEY     = st.secrets["FIREWORKS_API_KEY"]
FIREWORKS_EMBED_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
FIREWORKS_ENDPOINT    = "https://api.fireworks.ai/inference/v1/embeddings"

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_MODEL   = "gemini-2.5-flash"

BASE_DIR        = Path(__file__).parent
EMBEDDINGS_FILE = BASE_DIR / "Hajj_embeddings.json"
TOP_K           = 7
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
أنت مساعد متخصص في دليل الشروط الوقائية في المشاعر المقدسة (منى، مزدلفة، عرفة)، الصادر عن المديرية العامة للدفاع المدني السعودي.

مهمتك:
- الإجابة على أسئلة الحجاج وشركات الحج والعاملين في المشاعر المقدسة بشكل دقيق ومفيد.
- استند حصراً إلى المقاطع المسترجعة من الدليل المُرفقة بكل سؤال.
- إذا لم تجد إجابة في المقاطع المُقدَّمة، قل ذلك صراحةً ولا تخترع معلومات.
- أجب بنفس لغة السؤال (عربي أو إنجليزي).
- اجعل إجابتك واضحة ومنظمة، واستخدم النقاط عند الحاجة.\
"""

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="جارٍ تحميل قاعدة البيانات...")
def load_index():
    with open(EMBEDDINGS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    texts    = [c["text"]      for c in chunks]
    contexts = [c["context"]   for c in chunks]
    matrix   = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    # Pre-normalise rows so retrieval is a simple dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-9)
    return texts, contexts, matrix


@st.cache_resource
def load_gemini():
    return genai.Client(api_key=GEMINI_API_KEY)


# ── RAG functions ─────────────────────────────────────────────────────────────

def embed_query(text: str) -> list:
    for attempt in range(1, 4):
        resp = requests.post(
            FIREWORKS_ENDPOINT,
            headers={
                "Authorization": f"Bearer {FIREWORKS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": FIREWORKS_EMBED_MODEL, "input": [text]},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["data"][0]["embedding"]
        time.sleep(10 * attempt)
    raise RuntimeError("Failed to embed query.")


def retrieve(texts, contexts, matrix, query: str) -> list:
    q_emb = np.array(embed_query(query), dtype=np.float32)
    q_emb /= max(np.linalg.norm(q_emb), 1e-9)
    scores = matrix @ q_emb
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [
        {"text": texts[i], "context": contexts[i], "score": round(float(scores[i]), 4)}
        for i in top_idx
    ]


def generate_answer(gemini_client, question: str, chunks: list) -> str:
    context_block = "\n".join(
        f"[{i}] ({c['context']})\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    prompt = (
        f"السياق المسترجع من دليل الشروط الوقائية:\n"
        f"{'─' * 60}\n"
        f"{context_block}\n"
        f"{'─' * 60}\n\n"
        f"السؤال: {question}\n\nالإجابة:"
    )
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="دليل الشروط الوقائية — المشاعر المقدسة",
    page_icon="🕋",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* RTL for Arabic content */
    .rtl { direction: rtl; text-align: right; }

    /* Chat bubbles */
    .user-bubble {
        background: #1a5c38;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0;
        direction: rtl;
        text-align: right;
        max-width: 85%;
        margin-left: auto;
    }
    .assistant-bubble {
        background: #f0f2f6;
        color: #1a1a1a;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 0;
        direction: rtl;
        text-align: right;
        max-width: 92%;
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        background: #e8f4ea;
        color: #1a5c38;
        border: 1px solid #b8dfc0;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75em;
        margin: 2px;
        direction: rtl;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rtl">
    <h1>🕋 دليل الشروط الوقائية في المشاعر المقدسة</h1>
    <p style="color:#555;">نظام استجابة ذكي مبني على دليل الدفاع المدني السعودي لمشاعر منى ومزدلفة وعرفة</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load resources ────────────────────────────────────────────────────────────
texts, contexts, matrix = load_index()
gemini_client = load_gemini()

st.caption(f"✅ قاعدة المعرفة جاهزة — {len(texts)} مقطع | النموذج: {GEMINI_MODEL}")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Suggested questions ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<div class="rtl"><b>💡 أسئلة مقترحة:</b></div>', unsafe_allow_html=True)
    suggestions = [
        "ما هي اشتراطات طفايات الحريق في مخيمات الحجاج؟",
        "ما المواد الممنوع استخدامها في المشاعر المقدسة؟",
        "ما مهام مختص السلامة في المخيم؟",
        "ما اشتراطات مخارج الطوارئ في مخيمات عرفة؟",
        "What are the electrical safety rules in pilgrim camps?",
    ]
    cols = st.columns(1)
    for s in suggestions:
        if st.button(s, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": s})
            st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-bubble">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">[{i+1}] {s["context"]} ({s["score"]:.0%})</span>'
                for i, s in enumerate(msg["sources"])
            )
            st.markdown(
                f'<div class="rtl" style="margin-top:4px;">{chips}</div>',
                unsafe_allow_html=True,
            )

# ── Input ─────────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "سؤالك",
            placeholder="اكتب سؤالك هنا... / Type your question here...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("إرسال ➤", use_container_width=True)

if submitted and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🔍 جارٍ البحث وتوليد الإجابة..."):
        try:
            chunks  = retrieve(texts, contexts, matrix, question)
            answer  = generate_answer(gemini_client, question, chunks)
        except Exception as e:
            answer  = f"❌ حدث خطأ: {e}"
            chunks  = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })
    st.rerun()

# ── Clear button ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ مسح المحادثة", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
