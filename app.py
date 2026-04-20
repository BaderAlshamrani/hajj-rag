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

FIREWORKS_CHAT_API_KEY = "fw_PZrxeP3sQqT7QEVr7FR1sf"
FIREWORKS_CHAT_MODEL   = "accounts/fireworks/models/gpt-oss-20b"
FIREWORKS_CHAT_URL     = "https://api.fireworks.ai/inference/v1/chat/completions"

BASE_DIR        = Path(__file__).parent
EMBEDDINGS_FILE = BASE_DIR / "Hajj_embeddings.json"
TOP_K           = 5
CHUNK_MAX_CHARS = 800
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
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
/* ── CSS Variables ── */
:root {
    --bg:           #F9FAFB;
    --bg-surface:   #FFFFFF;
    --bg-card:      rgba(255,255,255,0.4);
    --bg-card-h:    rgba(255,255,255,0.75);
    --border:       #E5E7EB;
    --border-h:     #34D399;
    --emerald:      #34D399;
    --teal:         #14B8A6;
    --emerald-glow: rgba(52,211,153,0.22);
    --text:         #111827;
    --text-2:       #6B7280;
    --text-3:       #9CA3AF;
    --radius-lg:    24px;
    --radius-pill:  999px;
    --radius-md:    14px;
    --radius-sm:    10px;
    --shadow:       0 4px 20px rgba(0,0,0,0.06);
    --shadow-card:  0 2px 12px rgba(0,0,0,0.04), 0 1px 3px rgba(0,0,0,0.06);
    --shadow-lift:  0 12px 32px rgba(52,211,153,0.28);
    --font-ar:      'IBM Plex Sans Arabic', sans-serif;
    --font-en:      'Inter', sans-serif;
    --transition:   0.3s ease-in-out;
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
    background-image:
        radial-gradient(ellipse 80% 50% at 10% -10%, rgba(52,211,153,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 110%, rgba(20,184,166,0.05) 0%, transparent 60%);
}

/* subtle dot-grid overlay */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: radial-gradient(circle at 1px 1px, rgba(0,0,0,0.04) 1px, transparent 0);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

/* ── Block container ── */
.block-container {
    max-width: 800px !important;
    padding: 0 28px 80px !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.18); }

/* ── Keyframes ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.45; transform: scale(0.82); }
}

/* ──────────────────────────────────────────────
   HEADER
────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 56px 24px 32px;
    animation: fadeIn 0.8s ease;
}
.kaaba-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 76px; height: 76px;
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(20,184,166,0.08));
    border: 1px solid rgba(52,211,153,0.22);
    border-radius: 50%;
    font-size: 36px;
    margin-bottom: 24px;
    box-shadow: 0 0 0 8px rgba(52,211,153,0.06), 0 4px 24px rgba(52,211,153,0.14);
    transition: var(--transition);
}
.kaaba-wrap:hover {
    box-shadow: 0 0 0 12px rgba(52,211,153,0.08), 0 6px 32px rgba(52,211,153,0.2);
    transform: translateY(-2px);
}
.app-header h1 {
    font-family: var(--font-ar);
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 10px;
    line-height: 1.4;
    direction: rtl;
    letter-spacing: -0.01em;
}
.app-header p {
    color: var(--text-2);
    font-family: var(--font-ar);
    font-size: 0.9rem;
    margin: 0 0 24px;
    direction: rtl;
    line-height: 1.8;
    max-width: 580px;
    margin-left: auto;
    margin-right: auto;
}
.status-pill { display: none !important; }
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
    animation: pulse 2.4s ease-in-out infinite;
}
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 8px 0 36px;
    opacity: 0.7;
}

/* ──────────────────────────────────────────────
   SUGGESTION CHIPS  (Glassmorphism)
────────────────────────────────────────────── */
.sug-label {
    font-size: 0.72rem;
    font-family: var(--font-en);
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-3);
    text-align: right;
    direction: rtl;
    margin-bottom: 12px;
    font-weight: 600;
}

/* Override Streamlit buttons used as suggestion chips */
.stButton > button {
    background: rgba(255,255,255,0.4) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-2) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-ar) !important;
    font-size: 0.9rem !important;
    padding: 13px 20px !important;
    text-align: right !important;
    direction: rtl !important;
    width: 100%;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.6 !important;
    transition: all var(--transition) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: var(--shadow-card) !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.75) !important;
    border-color: var(--emerald) !important;
    color: var(--text) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 28px rgba(52,211,153,0.16), var(--shadow-card) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: var(--shadow-card) !important;
}

/* ──────────────────────────────────────────────
   CHAT MESSAGES
────────────────────────────────────────────── */
.msg-wrap {
    animation: fadeInUp 0.38s ease;
    margin-bottom: 24px;
}

/* User */
.msg-user {
    display: flex;
    justify-content: flex-end;
}
.msg-user .bubble {
    background: linear-gradient(135deg, var(--emerald) 0%, var(--teal) 100%);
    color: #fff;
    padding: 14px 20px;
    border-radius: 22px 22px 6px 22px;
    max-width: 76%;
    direction: rtl;
    text-align: right;
    font-family: var(--font-ar);
    font-size: 0.97rem;
    line-height: 1.8;
    box-shadow: 0 6px 24px rgba(52,211,153,0.28);
    word-break: break-word;
    font-weight: 500;
}

/* Assistant */
.msg-assistant {
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.bot-avatar {
    flex-shrink: 0;
    width: 40px; height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--emerald), var(--teal));
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 16px rgba(52,211,153,0.28);
    border: 1.5px solid rgba(52,211,153,0.25);
}
.msg-assistant .bubble {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 16px 22px;
    border-radius: 6px 22px 22px 22px;
    max-width: 87%;
    direction: rtl;
    text-align: right;
    font-family: var(--font-ar);
    font-size: 0.97rem;
    line-height: 2;
    box-shadow: var(--shadow);
    word-break: break-word;
    transition: box-shadow var(--transition);
}
.msg-assistant .bubble:hover {
    box-shadow: 0 6px 28px rgba(0,0,0,0.09);
}
.msg-assistant .bubble strong { color: #059669; }
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
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.2);
    color: #059669;
    border-radius: 30px;
    padding: 2px 12px;
    font-size: 0.7rem;
    font-family: var(--font-en);
    letter-spacing: 0.2px;
    transition: background var(--transition);
    font-weight: 500;
}
.src-chip:hover { background: rgba(52,211,153,0.15); }

/* ──────────────────────────────────────────────
   INPUT FORM  (Floating pill bar)
────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-pill) !important;
    padding: 10px 12px 10px 16px !important;
    box-shadow: 0 4px 32px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    margin-top: 32px !important;
    transition: box-shadow var(--transition), border-color var(--transition) !important;
}
[data-testid="stForm"]:focus-within {
    border-color: var(--emerald) !important;
    box-shadow: 0 0 0 4px rgba(52,211,153,0.12), 0 4px 32px rgba(0,0,0,0.08) !important;
}

.stTextInput > div > div > input,
[data-baseweb="base-input"] input,
[data-baseweb="input"] input {
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    color: var(--text) !important;
    font-family: var(--font-ar) !important;
    font-size: 1rem !important;
    direction: rtl !important;
    text-align: right !important;
    padding: 8px 4px !important;
    caret-color: var(--emerald) !important;
    box-shadow: none !important;
    transition: all var(--transition) !important;
}
.stTextInput > div > div > input:focus,
[data-baseweb="base-input"] input:focus {
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder,
[data-baseweb="base-input"] input::placeholder {
    color: var(--text-3) !important;
    font-size: 0.92rem !important;
}
[data-testid="stTextInputRootElement"],
[data-baseweb="base-input"],
[data-baseweb="input"] {
    background: #FFFFFF !important;
    border: none !important;
    box-shadow: none !important;
}

/* Send button — Emerald-400 → Teal-500 gradient with hover lift */
[data-testid="stForm"] .stFormSubmitButton > button {
    background: linear-gradient(135deg, #34D399 0%, #14B8A6 100%) !important;
    border: none !important;
    color: #fff !important;
    border-radius: var(--radius-pill) !important;
    font-family: var(--font-ar) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 10px 26px !important;
    height: 44px !important;
    box-shadow: 0 4px 18px rgba(52,211,153,0.32) !important;
    transition: all var(--transition) !important;
    white-space: nowrap !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stForm"] .stFormSubmitButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--shadow-lift) !important;
    background: linear-gradient(135deg, #2dd4a0 0%, #0fa89a 100%) !important;
}
[data-testid="stForm"] .stFormSubmitButton > button:active {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(52,211,153,0.24) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--emerald) !important; }
[data-testid="stSpinner"] p {
    color: var(--text-2) !important;
    font-family: var(--font-ar) !important;
    direction: rtl !important;
    font-size: 0.9rem !important;
}

/* ── Caption & stray elements ── */
.stCaption { display: none !important; }
[data-testid="stHorizontalBlock"] { gap: 12px !important; }

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
    border-radius: var(--radius-pill) !important;
    padding: 7px 18px !important;
    height: auto !important;
    width: auto !important;
    box-shadow: none !important;
    transition: all var(--transition) !important;
}
.clear-wrap .stButton > button:hover {
    border-color: rgba(239,68,68,0.35) !important;
    color: #ef4444 !important;
    background: rgba(239,68,68,0.05) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ──────────────────────────────────────────────
   MOBILE  (≤ 640px)
────────────────────────────────────────────── */
@media (max-width: 640px) {
    .block-container {
        padding: 0 14px 100px !important;
    }

    /* Header */
    .app-header {
        padding: 32px 8px 20px !important;
    }
    .kaaba-wrap {
        width: 60px !important; height: 60px !important;
        font-size: 28px !important;
        margin-bottom: 16px !important;
    }
    .app-header h1 {
        font-size: 1.25rem !important;
        line-height: 1.5 !important;
    }
    .app-header p {
        font-size: 0.82rem !important;
        line-height: 1.7 !important;
    }

    /* Suggestion chips — stack vertically */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 8px !important;
    }
    .stButton > button {
        font-size: 0.85rem !important;
        padding: 11px 16px !important;
    }

    /* Chat bubbles */
    .msg-user .bubble {
        max-width: 90% !important;
        font-size: 0.92rem !important;
        padding: 12px 16px !important;
    }
    .msg-assistant .bubble {
        max-width: 95% !important;
        font-size: 0.92rem !important;
        padding: 12px 16px !important;
    }
    .bot-avatar {
        width: 32px !important; height: 32px !important;
        font-size: 16px !important;
    }
    .msg-assistant { gap: 8px !important; }

    /* Input form — stick to bottom */
    [data-testid="stForm"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        margin: 0 !important;
        border-radius: 20px 20px 0 0 !important;
        padding: 10px 12px !important;
        z-index: 999 !important;
        border-left: none !important;
        border-right: none !important;
        border-bottom: none !important;
        box-shadow: 0 -4px 24px rgba(0,0,0,0.10) !important;
    }
    [data-testid="stForm"]:focus-within {
        border-radius: 20px 20px 0 0 !important;
    }

    /* Send button — smaller on mobile */
    [data-testid="stForm"] .stFormSubmitButton > button {
        padding: 10px 18px !important;
        font-size: 0.88rem !important;
        height: 40px !important;
    }

    /* Input text */
    .stTextInput > div > div > input,
    [data-baseweb="base-input"] input {
        font-size: 0.95rem !important;
    }

    /* Source chips — smaller */
    .src-chip {
        font-size: 0.65rem !important;
        padding: 2px 9px !important;
    }
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




# ── Helpers ───────────────────────────────────────────────────────────────────

def md_to_html(text: str) -> str:
    """Convert basic markdown + LaTeX to safe HTML. No asterisk bullets."""
    # ── Strip LaTeX math BEFORE html-escaping ──────────────────────────────
    # \times → ×
    text = re.sub(r'\\times', '×', text)
    # \text{ ... } → contents only
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    # \cdot → ·
    text = re.sub(r'\\cdot', '·', text)
    # \frac{a}{b} → a/b
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', text)
    # \sqrt{x} → √x
    text = re.sub(r'\\sqrt\{([^}]*)\}', r'√\1', text)
    # strip remaining \cmd sequences (e.g. \left, \right, \approx …)
    text = re.sub(r'\\[a-zA-Z]+\b\*?', '', text)
    # block math  \[ ... \]  — unwrap, keeping inner content on its own line
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'\n\1\n', text, flags=re.DOTALL)
    # inline math  \( ... \)  — unwrap
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'\1', text, flags=re.DOTALL)
    # lone { } braces left over
    text = re.sub(r'[{}]', '', text)

    # ── HTML escape then convert markdown ──────────────────────────────────
    text = _html.escape(text)
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


def generate_answer(question: str, chunks: list) -> str:
    context_block = "\n".join(
        f"[{i}] ({c['context']})\n{c['text'][:CHUNK_MAX_CHARS]}" for i, c in enumerate(chunks, 1)
    )
    user_content = (
        f"السياق المسترجع من دليل الشروط الوقائية:\n"
        f"{'─' * 60}\n"
        f"{context_block}\n"
        f"{'─' * 60}\n\n"
        f"السؤال: {question}\n\nالإجابة:"
    )
    resp = requests.post(
        FIREWORKS_CHAT_URL,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {FIREWORKS_CHAT_API_KEY}",
        },
        data=json.dumps({
            "model": FIREWORKS_CHAT_MODEL,
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        }),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


# ── Load resources ────────────────────────────────────────────────────────────
texts, contexts, matrix = load_index()

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
                    answer = generate_answer(s, chunks)
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
            answer = generate_answer(question, chunks)
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
