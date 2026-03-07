"""
Hajj Safety Guide — RAG Streamlit App
"""

import html as _html
import json
import re
import time
import requests
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from pathlib import Path

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="دليل الشروط الوقائية — المشاعر المقدسة",
    page_icon="🕋",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Config ────────────────────────────────────────────────────────────────────
FIREWORKS_API_KEY     = st.secrets["FIREWORKS_API_KEY"]
FIREWORKS_EMBED_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
FIREWORKS_ENDPOINT    = "https://api.fireworks.ai/inference/v1/embeddings"

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL    = "openai/gpt-oss-120b"

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
- اجعل إجابتك واضحة ومنظمة، واستخدم التعداد بالأرقام (١، ٢، ٣ أو 1. 2. 3.) ولا تستخدم النجمة (*) في القوائم.\
"""

# ── Inject Google Fonts + full design system ──────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── CSS Variables ── */
:root {
    --bg:           #0b0f1a;
    --bg-surface:   #111827;
    --bg-card:      rgba(255,255,255,0.035);
    --bg-card-h:    rgba(255,255,255,0.06);
    --border:       rgba(255,255,255,0.07);
    --border-h:     rgba(0,184,148,0.45);
    --green:        #00b894;
    --green-dk:     #009e7f;
    --green-glow:   rgba(0,184,148,0.18);
    --gold:         #f0c040;
    --gold-dim:     rgba(240,192,64,0.12);
    --text:         #eaeaea;
    --text-2:       #8b9ab0;
    --text-3:       #3d4d60;
    --radius-lg:    20px;
    --radius-md:    14px;
    --radius-sm:    10px;
    --shadow:       0 8px 32px rgba(0,0,0,0.45);
    --font-ar:      'Noto Naskh Arabic', serif;
    --font-en:      'Inter', sans-serif;
}

/* ── Erase Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"],
.stDeployButton { display: none !important; }

/* ── App shell ── */
.stApp {
    background: var(--bg) !important;
    font-family: var(--font-ar);
    color: var(--text);
    /* subtle ambient glow */
    background-image:
        radial-gradient(ellipse 90% 55% at 15% -5%,  rgba(0,184,148,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 70% 45% at 85% 105%, rgba(240,192,64,0.05) 0%, transparent 55%);
}

/* dot-grid overlay */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: radial-gradient(circle at 1px 1px, rgba(255,255,255,0.025) 1px, transparent 0);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
}

/* ── Block container ── */
.block-container {
    max-width: 780px !important;
    padding: 0 20px 60px !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.16); }

/* ── Keyframes ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.85); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ──────────────────────────────────────────────
   HEADER
────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 52px 24px 28px;
    animation: fadeIn 0.9s ease;
}
.kaaba-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 72px; height: 72px;
    background: linear-gradient(135deg, rgba(0,184,148,0.15), rgba(240,192,64,0.08));
    border: 1px solid rgba(0,184,148,0.25);
    border-radius: 50%;
    font-size: 36px;
    margin-bottom: 20px;
    box-shadow: 0 0 40px rgba(0,184,148,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
}
.app-header h1 {
    font-family: var(--font-ar);
    font-size: 1.85rem;
    font-weight: 700;
    background: linear-gradient(120deg, #ffffff 0%, #a8f0dc 45%, #f0c040 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 6s ease infinite;
    margin: 0 0 8px;
    line-height: 1.35;
    direction: rtl;
}
.app-header p {
    color: var(--text-2);
    font-size: 0.88rem;
    margin: 0 0 20px;
    direction: rtl;
}
.status-pill { display: none !important; }
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
    animation: pulse 2.2s ease-in-out infinite;
}
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 6px 0 28px;
}

/* ──────────────────────────────────────────────
   SUGGESTION CHIPS
────────────────────────────────────────────── */
.sug-label {
    font-size: 0.75rem;
    font-family: var(--font-en);
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-3);
    text-align: right;
    direction: rtl;
    margin-bottom: 10px;
}

/* Override Streamlit buttons used as suggestion chips */
.stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-2) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-ar) !important;
    font-size: 0.9rem !important;
    padding: 11px 18px !important;
    text-align: right !important;
    direction: rtl !important;
    width: 100%;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.5 !important;
    transition: all 0.22s ease !important;
    backdrop-filter: blur(6px) !important;
}
.stButton > button:hover {
    background: var(--bg-card-h) !important;
    border-color: var(--green) !important;
    color: var(--green) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px var(--green-glow) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ──────────────────────────────────────────────
   CHAT MESSAGES
────────────────────────────────────────────── */
.msg-wrap {
    animation: fadeInUp 0.35s ease;
    margin-bottom: 20px;
}

/* User */
.msg-user {
    display: flex;
    justify-content: flex-end;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #00b894 0%, #009e7f 100%);
    color: #fff;
    padding: 14px 20px;
    border-radius: 22px 22px 5px 22px;
    max-width: 78%;
    direction: rtl;
    text-align: right;
    font-family: var(--font-ar);
    font-size: 0.97rem;
    line-height: 1.75;
    box-shadow: 0 6px 24px rgba(0,184,148,0.28);
    word-break: break-word;
}

/* Assistant */
.msg-assistant {
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.bot-avatar {
    flex-shrink: 0;
    width: 38px; height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00b894, #009e7f);
    display: flex; align-items: center; justify-content: center;
    font-size: 19px;
    box-shadow: 0 0 18px rgba(0,184,148,0.35);
    border: 1px solid rgba(0,184,148,0.3);
}
.msg-assistant .bubble {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 15px 20px;
    border-radius: 5px 22px 22px 22px;
    max-width: 87%;
    direction: rtl;
    text-align: right;
    font-family: var(--font-ar);
    font-size: 0.97rem;
    line-height: 2;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: var(--shadow);
    word-break: break-word;
}
.msg-assistant .bubble strong { color: #a8f0dc; }
.msg-assistant .bubble br + br { margin-top: 4px; }

/* Sources */
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 12px;
    justify-content: flex-end;
    padding-top: 12px;
    border-top: 1px solid var(--border);
}
.src-chip {
    background: var(--gold-dim);
    border: 1px solid rgba(240,192,64,0.18);
    color: var(--gold);
    border-radius: 30px;
    padding: 2px 11px;
    font-size: 0.7rem;
    font-family: var(--font-en);
    letter-spacing: 0.2px;
    transition: background 0.2s;
}
.src-chip:hover { background: rgba(240,192,64,0.22); }

/* ──────────────────────────────────────────────
   INPUT FORM
────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 14px 16px !important;
    box-shadow: 0 -2px 40px rgba(0,0,0,0.35) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    margin-top: 28px !important;
}
[data-testid="stForm"]:focus-within {
    border-color: var(--border-h) !important;
    box-shadow: 0 0 0 3px var(--green-glow), 0 -2px 40px rgba(0,0,0,0.35) !important;
}

.stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-family: var(--font-ar) !important;
    font-size: 1rem !important;
    direction: rtl !important;
    text-align: right !important;
    padding: 8px 4px !important;
    caret-color: var(--green) !important;
    box-shadow: none !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus {
    border-bottom-color: var(--green) !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--text-3) !important;
    font-size: 0.9rem !important;
}
[data-testid="stTextInputRootElement"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Send button */
[data-testid="stForm"] .stFormSubmitButton > button {
    background: linear-gradient(135deg, #00b894, #009e7f) !important;
    border: none !important;
    color: #fff !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-ar) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    height: 44px !important;
    box-shadow: 0 4px 18px var(--green-glow) !important;
    transition: all 0.22s ease !important;
    white-space: nowrap !important;
}
[data-testid="stForm"] .stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,184,148,0.38) !important;
}
[data-testid="stForm"] .stFormSubmitButton > button:active {
    transform: translateY(0) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--green) !important; }
[data-testid="stSpinner"] p {
    color: var(--text-2) !important;
    font-family: var(--font-ar) !important;
    direction: rtl !important;
    font-size: 0.9rem !important;
}

/* ── Caption & stray elements ── */
.stCaption { display: none !important; }
[data-testid="stHorizontalBlock"] { gap: 10px !important; }

/* Hide "Press Enter to submit form" hint (Streamlit injects it in form) */
[data-testid="stForm"] [data-testid="stMarkdown"] { display: none !important; }
[data-testid="stForm"] p { display: none !important; }
[data-testid="stForm"] small { display: none !important; }
[data-testid="stForm"] [class*="instruction"] { display: none !important; }
/* Keep submit button content visible (إرسال) */
[data-testid="stForm"] .stFormSubmitButton p,
[data-testid="stForm"] .stFormSubmitButton small,
[data-testid="stForm"] .stFormSubmitButton button * { display: block !important; visibility: visible !important; }

/* ── Clear button ── */
.clear-wrap .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-3) !important;
    font-size: 0.82rem !important;
    border-radius: var(--radius-sm) !important;
    padding: 6px 14px !important;
    height: auto !important;
    width: auto !important;
}
.clear-wrap .stButton > button:hover {
    border-color: rgba(239,68,68,0.4) !important;
    color: #f87171 !important;
    background: rgba(239,68,68,0.06) !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="جارٍ تحميل قاعدة البيانات...")
def load_index():
    with open(EMBEDDINGS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    chunks   = data["chunks"]
    texts    = [c["text"]    for c in chunks]
    contexts = [c["context"] for c in chunks]
    matrix   = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms    = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix   = matrix / np.maximum(norms, 1e-9)
    return texts, contexts, matrix


@st.cache_resource
def load_groq():
    return Groq(api_key=GROQ_API_KEY)


# ── Helpers ───────────────────────────────────────────────────────────────────

def md_to_html(text: str) -> str:
    """Convert basic Gemini markdown to safe HTML for injection. No asterisk bullets."""
    text = _html.escape(text)
    # Convert markdown list lines (starting with * or -) to bullet • so asterisk is not shown
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^\s*[\*\-]\s+", line):
            lines[i] = re.sub(r"^\s*[\*\-]\s+", "• ", line)
    text = "\n".join(lines)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
    text = text.replace("\n", "<br>")
    return text


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
    scores  = matrix @ q_emb
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [
        {"text": texts[i], "context": contexts[i], "score": round(float(scores[i]), 4)}
        for i in top_idx
    ]


def generate_answer(groq_client: Groq, question: str, chunks: list) -> str:
    context_block = "\n".join(
        f"[{i}] ({c['context']})\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    user_content = (
        f"السياق المسترجع من دليل الشروط الوقائية:\n"
        f"{'─' * 60}\n"
        f"{context_block}\n"
        f"{'─' * 60}\n\n"
        f"السؤال: {question}\n\nالإجابة:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None,
    )
    parts = []
    for chunk in completion:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                parts.append(delta.content)
    return "".join(parts) if parts else ""


# ── Load resources ────────────────────────────────────────────────────────────
texts, contexts, matrix = load_index()
groq_client = load_groq()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <div class="kaaba-wrap">🕋</div>
    <h1>دليل الشروط الوقائية في المشاعر المقدسة</h1>
    <p>دليل رقمي ذكي للإجابة على استفسارات الحماية من الحريق، وأنظمة الطبخ، والسلامة الكهربائية، ومهام مختصي السلامة في المشاعر المقدسة وفق أحدث المعايير التنظيمية المعتمدة لموسم الحج.</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Suggested questions (click to ask and get answer) ───────────────────────────
if not st.session_state.messages:
    st.markdown('<div class="sug-label">💡 أسئلة مقترحة</div>', unsafe_allow_html=True)
    suggestions = [
        "ما هي اشتراطات طفايات الحريق في مخيمات الحجاج؟",
        "ما المواد الممنوع استخدامها في المشاعر المقدسة؟",
        "ما اشتراطات مخارج الطوارئ في مخيمات عرفة؟",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{hash(s)}"):
            st.session_state.messages.append({"role": "user", "content": s})
            with st.spinner("جارٍ البحث وتوليد الإجابة..."):
                try:
                    chunks = retrieve(texts, contexts, matrix, s)
                    answer = generate_answer(groq_client, s, chunks)
                except Exception as e:
                    answer = f"❌ حدث خطأ: {e}"
                    chunks = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": chunks,
            })
            st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        safe_text = _html.escape(msg["content"])
        st.markdown(f"""
<div class="msg-wrap">
  <div class="msg-user">
    <div class="bubble">{safe_text}</div>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        body_html = md_to_html(msg["content"])
        st.markdown(f"""
<div class="msg-wrap">
  <div class="msg-assistant">
    <div class="bot-avatar">🤖</div>
    <div class="bubble">
      {body_html}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Hide "Press Enter to submit form" via script (runs in page context) ────────
_hide_form_hint_js = """
<script>
(function() {
  var doc = (window.parent && window.parent.document) ? window.parent.document : document;
  function hideHint() {
    var root = doc.querySelector('[data-testid="stForm"]') || doc.body;
    var walk = function(el) {
      if (!el) return;
      if (el.nodeType === 3) {
        var t = (el.textContent || '').trim();
        if (/Press\\s+Enter|submit\\s+form/i.test(t) && el.parentNode) {
          var p = el.parentNode;
          if (p && p.style) p.style.setProperty('display', 'none', 'important');
        }
        return;
      }
      if (el.nodeType === 1 && el.childNodes) {
        for (var i = 0; i < el.childNodes.length; i++) walk(el.childNodes[i]);
      }
    };
    walk(root);
  }
  if (doc.readyState === 'complete') hideHint();
  else doc.addEventListener('DOMContentLoaded', hideHint);
  [300, 800, 1500, 2500].forEach(function(ms) { setTimeout(hideHint, ms); });
})();
</script>
"""
components.html(_hide_form_hint_js, height=0)

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True, enter_to_submit=False):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "سؤالك",
            placeholder="اكتب سؤالك هنا...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("إرسال", use_container_width=True)

if submitted and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("جارٍ البحث وتوليد الإجابة..."):
        try:
            chunks = retrieve(texts, contexts, matrix, question)
            answer = generate_answer(groq_client, question, chunks)
        except Exception as e:
            answer = f"❌ حدث خطأ: {e}"
            chunks = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })
    st.rerun()

# ── Clear button ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([4, 2, 4])
    with c2:
        st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
        if st.button("🗑 مسح المحادثة", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
