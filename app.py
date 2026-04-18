import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import io
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FarmVoice AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM  —  Black × White × Green editorial
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --black:       #0a0a0a;
    --off-black:   #111111;
    --dark:        #1a1a1a;
    --mid:         #2a2a2a;
    --border:      #2e2e2e;
    --muted:       #555555;
    --light-muted: #888888;
    --off-white:   #f0ede8;
    --white:       #ffffff;
    --green:       #52b788;
    --green-dim:   #2d6a4f;
    --green-glow:  rgba(82,183,136,0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--black) !important;
    color: var(--off-white);
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--off-black); }
::-webkit-scrollbar-thumb { background: var(--green-dim); border-radius: 2px; }

/* ══ HERO ══════════════════════════════════════════════════════════════════ */
.hero {
    border-bottom: 1px solid var(--border);
    padding: 48px 0 40px 0;
    margin-bottom: 40px;
    position: relative;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: var(--green);
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3.5rem, 8vw, 7rem);
    color: var(--white);
    line-height: 0.9;
    margin: 0 0 20px 0;
    letter-spacing: 2px;
}
.hero-title span { color: var(--green); }
.hero-sub {
    font-size: 0.95rem;
    color: var(--light-muted);
    font-weight: 300;
    max-width: 520px;
    line-height: 1.6;
}
.hero-rule {
    position: absolute;
    right: 0; top: 48px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 8rem;
    color: var(--border);
    line-height: 1;
    user-select: none;
    pointer-events: none;
}

/* ══ STAT STRIP ════════════════════════════════════════════════════════════ */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 40px;
}
.stat-cell {
    background: var(--off-black);
    padding: 20px 24px;
    text-align: center;
}
.stat-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: var(--white);
    line-height: 1;
    display: block;
}
.stat-num.green { color: var(--green); }
.stat-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 6px;
    display: block;
}

/* ══ TABS ═══════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 0 !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 14px 24px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important;
}

/* ══ PANELS ══════════════════════════════════════════════════════════════════ */
.panel {
    background: var(--off-black);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 32px;
    margin-bottom: 20px;
}
.panel-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    color: var(--white);
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.panel-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 24px;
    line-height: 1.5;
}

/* ══ MIC INSTRUCTION PANEL ══════════════════════════════════════════════════ */
.mic-panel {
    background: var(--dark);
    border: 1px solid var(--green-dim);
    border-left: 3px solid var(--green);
    border-radius: 4px;
    padding: 20px 24px;
    margin: 16px 0 24px 0;
}
.mic-panel ol {
    margin: 0;
    padding-left: 20px;
    color: var(--light-muted);
    font-size: 0.88rem;
    line-height: 2.1;
}
.mic-panel ol li strong { color: var(--white); }
.mic-panel ol li em { color: var(--green); font-style: normal; }

/* ══ RESULT BOX ══════════════════════════════════════════════════════════════ */
.result-wrap {
    background: var(--dark);
    border: 1px solid var(--border);
    border-top: 3px solid var(--green);
    border-radius: 4px;
    padding: 40px;
    text-align: center;
    margin: 24px 0;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, var(--green-glow) 0%, transparent 70%);
    pointer-events: none;
}
.result-emoji { font-size: 4rem; display: block; margin-bottom: 8px; }
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: var(--green);
    text-transform: uppercase;
    margin-bottom: 8px;
}
.result-crop {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: var(--white);
    letter-spacing: 3px;
    line-height: 1;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.result-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--light-muted);
}

/* ══ ALT ROW ═════════════════════════════════════════════════════════════════ */
.alt-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}
.alt-bar {
    height: 4px;
    background: var(--green);
    border-radius: 2px;
    flex-shrink: 0;
}

/* ══ GROQ STATUS TAG ════════════════════════════════════════════════════════ */
.groq-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--dark);
    border: 1px solid var(--green-dim);
    border-radius: 2px;
    padding: 4px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--green);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.groq-dot {
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:0.3; }
}

/* ══ SIDEBAR ══════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--off-black) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--off-white) !important; }

/* ══ BUTTONS ══════════════════════════════════════════════════════════════════ */
.stButton > button {
    background: var(--white) !important;
    color: var(--black) !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 14px 28px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--green) !important;
    color: var(--black) !important;
    transform: translateY(-1px) !important;
}

/* ══ INPUTS ═══════════════════════════════════════════════════════════════════ */
.stTextArea textarea, .stTextInput input, .stNumberInput input {
    background: var(--dark) !important;
    color: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 1px var(--green) !important;
}
label, .stMarkdown p, .stMarkdown li { color: var(--off-white) !important; }
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3 { color: var(--white) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

model, le = load_model()
explainer  = get_explainer(model)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CROP_EMOJI = {
    'rice':'🌾','wheat':'🌾','maize':'🌽','chickpea':'🫘',
    'kidneybeans':'🫘','pigeonpeas':'🫘','mothbeans':'🫘',
    'mungbean':'🫘','blackgram':'🫘','lentil':'🫘',
    'pomegranate':'🍎','banana':'🍌','mango':'🥭',
    'grapes':'🍇','watermelon':'🍉','muskmelon':'🍈',
    'apple':'🍎','orange':'🍊','papaya':'🍈',
    'coconut':'🥥','cotton':'🌿','jute':'🌿',
}
FEATURE_LABELS = {
    'N':'Nitrogen (N)','P':'Phosphorus (P)','K':'Potassium (K)',
    'temperature':'Temperature','humidity':'Humidity',
    'ph':'Soil pH','rainfall':'Rainfall',
}
FEATURE_TIPS = {
    'N':'Essential for leaf growth. Higher = lush green crops.',
    'P':'Root & flower development. Legumes need more.',
    'K':'Disease resistance & fruit quality.',
    'temperature':'Average seasonal temperature in °C.',
    'humidity':'Average relative humidity in your region.',
    'ph':'Soil acidity. 7 = neutral, <7 = acidic, >7 = alkaline.',
    'rainfall':'Average annual rainfall in mm.',
}
COLS = ['N','P','K','temperature','humidity','ph','rainfall']


# ══════════════════════════════════════════════════════════════════════════════
# GROQ WHISPER  — 100% server-side, no browser mic required
# Free tier: 7,200 seconds/day · Accepts WAV, MP3, MP4, M4A, OGG, WEBM, FLAC
# ══════════════════════════════════════════════════════════════════════════════
def transcribe_with_groq(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    try:
        from groq import Groq
        api_key = (st.secrets.get("GROQ_API_KEY", "")
                   or os.environ.get("GROQ_API_KEY", ""))
        if not api_key:
            st.error("⚠️  Add GROQ_API_KEY to your Streamlit secrets. Get a free key at console.groq.com")
            return ""
        client = Groq(api_key=api_key)
        ext  = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
        mime = {
            "wav":"audio/wav","mp3":"audio/mpeg","mp4":"audio/mp4",
            "m4a":"audio/mp4","ogg":"audio/ogg","webm":"audio/webm",
            "flac":"audio/flac",
        }.get(ext, "audio/wav")
        result = client.audio.transcriptions.create(
            model          = "whisper-large-v3-turbo",
            file           = (filename, io.BytesIO(audio_bytes), mime),
            language       = "en",
            response_format= "text",
        )
        return (result if isinstance(result, str) else result.text).strip()
    except Exception as e:
        st.error(f"Groq error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# NLP PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_voice_input(text: str) -> dict:
    t = text.lower()
    return {
        'N': (100 if any(w in t for w in ['very fertile','very rich','high nitrogen','lots of manure'])
              else 75 if any(w in t for w in ['fertile','rich soil','good soil','dark soil','black soil'])
              else 25 if any(w in t for w in ['poor soil','low fertility','sandy','dry land'])
              else 50),
        'P': (80 if any(w in t for w in ['bean','pulse','legume','dal','chickpea'])
              else 30 if any(w in t for w in ['fruit','mango','banana','orchard'])
              else 50),
        'K': (70 if any(w in t for w in ['good harvest','healthy crop','strong plant'])
              else 25 if any(w in t for w in ['disease','weak','poor yield'])
              else 45),
        'temperature': (38 if any(w in t for w in ['very hot','extreme heat','scorching','desert'])
                        else 30 if any(w in t for w in ['hot','warm','summer','tropical'])
                        else 15 if any(w in t for w in ['cold','cool','winter','hill','mountain'])
                        else 22 if any(w in t for w in ['mild','moderate','pleasant'])
                        else 25),
        'humidity': (85 if any(w in t for w in ['very humid','coastal','very wet','paddy','rice field'])
                     else 70 if any(w in t for w in ['humid','wet','rainy season','monsoon'])
                     else 35 if any(w in t for w in ['dry','arid','low humidity'])
                     else 55),
        'ph': (7.8 if any(w in t for w in ['black soil','cotton soil','regur'])
               else 6.0 if any(w in t for w in ['red soil','laterite','acidic'])
               else 6.5 if any(w in t for w in ['sandy','light soil'])
               else 8.2 if any(w in t for w in ['alkaline','salt','saline'])
               else 6.8),
        'rainfall': (250 if any(w in t for w in ['heavy rain','floods','very rainy'])
                     else 140 if any(w in t for w in ['good rain','monsoon','rainy'])
                     else 50  if any(w in t for w in ['less rain','low rain','drought','dry season'])
                     else 90  if any(w in t for w in ['moderate rain','some rain'])
                     else 100),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT + EXPLAIN
# ══════════════════════════════════════════════════════════════════════════════
def predict_and_explain(features: dict):
    df    = pd.DataFrame([features])[COLS]
    enc   = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    crop  = le.inverse_transform([enc])[0]
    conf  = proba.max() * 100
    top3  = [(le.inverse_transform([i])[0], proba[i]*100)
             for i in np.argsort(proba)[::-1][:3]]
    sv = explainer.shap_values(df)
    sp = (sv[enc] if isinstance(sv, list)
          else sv[:,:,enc] if sv.ndim==3 else sv)
    shap_s = pd.Series(sp[0] if sp.ndim > 1 else sp, index=COLS)
    return crop, conf, top3, shap_s, df

def plain_english(crop, shap_s, features):
    lines = []
    for feat in shap_s.abs().sort_values(ascending=False).index[:3]:
        val  = shap_s[feat]
        lbl  = FEATURE_LABELS[feat]
        raw  = features[feat]
        unit = '°C' if feat=='temperature' else '%' if feat=='humidity' else ' mm' if feat=='rainfall' else ''
        if val > 0.01:
            lines.append(f"✅ **{lbl}** ({raw:.1f}{unit}) strongly favors **{crop}**")
        elif val < -0.01:
            lines.append(f"⚠️ **{lbl}** ({raw:.1f}{unit}) slightly reduces confidence — {crop} still best")
        else:
            lines.append(f"➡️ **{lbl}** ({raw:.1f}{unit}) neutral effect")
    return lines

def plot_shap(shap_s, crop):
    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor('#111')
    ax.set_facecolor('#111')
    colors = ['#52b788' if v>0 else '#ef4444' for v in shap_s.values]
    labels = [FEATURE_LABELS.get(f,f) for f in shap_s.index]
    bars   = ax.barh(labels, shap_s.values, color=colors, edgecolor='none', height=0.55)
    ax.axvline(x=0, color='#444', linewidth=1)
    ax.set_xlabel('SHAP Value', fontsize=9, color='#555')
    ax.set_title(f'WHY  {crop.upper()}', fontsize=13, fontweight='bold',
                 color='#fff', pad=14, fontfamily='monospace')
    for bar, val in zip(bars, shap_s.values):
        ax.text(val+(0.004 if val>=0 else -0.004), bar.get_y()+bar.get_height()/2,
                f'{val:+.3f}', va='center', ha='left' if val>=0 else 'right',
                fontsize=8, color='#aaa')
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.tick_params(axis='y', labelsize=9, colors='#ccc')
    ax.tick_params(axis='x', labelsize=8, colors='#555')
    ax.grid(axis='x', linestyle='--', alpha=0.15, color='#444')
    plt.tight_layout()
    return fig

def plot_importance():
    imp = pd.Series(model.feature_importances_, index=COLS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor('#111')
    ax.set_facecolor('#111')
    bars = ax.barh([FEATURE_LABELS[f] for f in imp.index], imp.values,
                   color='#ffffff', edgecolor='none', height=0.5)
    for bar, val in zip(bars, imp.values):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left', fontsize=8, color='#888')
    ax.set_xlabel('Importance', fontsize=9, color='#555')
    ax.set_title('GLOBAL FEATURE IMPORTANCE', fontsize=10,
                 color='#fff', pad=12, fontfamily='monospace', fontweight='bold')
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.tick_params(axis='y', labelsize=9, colors='#aaa')
    ax.tick_params(axis='x', labelsize=8, colors='#555')
    ax.grid(axis='x', linestyle='--', alpha=0.15, color='#444')
    plt.tight_layout()
    return fig

def render_result(crop, conf, top3, shap_s, features):
    emoji = CROP_EMOJI.get(crop,'🌱')
    st.markdown(f"""
    <div class="result-wrap">
        <span class="result-emoji">{emoji}</span>
        <div class="result-label">Recommended Crop</div>
        <div class="result-crop">{crop}</div>
        <div class="result-conf">Confidence: {conf:.1f}%</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**TOP 3 ALTERNATIVES**")
        for c, prob in top3:
            w = max(4, int(prob * 1.2))
            st.markdown(f"""
            <div class="alt-row">
                <span>{CROP_EMOJI.get(c,'🌱')}</span>
                <span style="color:#fff;font-weight:500;min-width:110px">{c.capitalize()}</span>
                <div class="alt-bar" style="width:{w}px;"></div>
                <span style="color:#888;font-size:0.85rem;font-family:monospace">{prob:.1f}%</span>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("**WHY THIS CROP**")
        for line in plain_english(crop, shap_s, features):
            st.markdown(line)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**SHAP FEATURE IMPACT**")
    st.caption("Green = supports this crop  ·  Red = works against it")
    fig = plot_shap(shap_s, crop)
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# HERO + STATS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-rule">🌾</div>
    <div class="hero-eyebrow">XAI · Agri-Tech · India · 2024</div>
    <div class="hero-title">FARM<span>VOICE</span><br>AI</div>
    <p class="hero-sub">Explainable crop advisory for Indian farmers.<br>
    Voice-powered via Groq Whisper · SHAP explainability · Random Forest.</p>
</div>

<div class="stat-strip">
    <div class="stat-cell">
        <span class="stat-num">22</span><span class="stat-lbl">Crops</span>
    </div>
    <div class="stat-cell">
        <span class="stat-num green">89%</span><span class="stat-lbl">Accuracy</span>
    </div>
    <div class="stat-cell">
        <span class="stat-num">7</span><span class="stat-lbl">Features</span>
    </div>
    <div class="stat-cell">
        <span class="stat-num green">XAI</span><span class="stat-lbl">SHAP</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌱 Farm Parameters")
    st.markdown("---")
    st.markdown("**Soil Nutrients**")
    N           = st.slider("Nitrogen (N) kg/ha",   0,   140, 60,  help=FEATURE_TIPS['N'])
    P           = st.slider("Phosphorus (P) kg/ha", 5,   145, 50,  help=FEATURE_TIPS['P'])
    K           = st.slider("Potassium (K) kg/ha",  5,   205, 50,  help=FEATURE_TIPS['K'])
    st.markdown("**Climate**")
    temperature = st.slider("Temperature (°C)",      5,   45,  25,  help=FEATURE_TIPS['temperature'])
    humidity    = st.slider("Humidity (%)",          20,  100, 60,  help=FEATURE_TIPS['humidity'])
    rainfall    = st.slider("Rainfall (mm)",         20,  300, 100, help=FEATURE_TIPS['rainfall'])
    st.markdown("**Soil**")
    ph          = st.slider("Soil pH",               3.5, 9.5, 6.5, step=0.1, help=FEATURE_TIPS['ph'])
    st.markdown("---")
    st.markdown("**FarmVoice AI**")
    st.markdown("Built by **Akash M S**")
    st.markdown("Presidency University, Bangalore")
    st.markdown("Random Forest · SHAP · Groq Whisper")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎙  Voice / Text", "⚙  Manual Input", "📊  Model Insights"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Voice via Groq Whisper upload
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Describe Your Farm</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-sub">Upload a voice recording — Groq Whisper transcribes it instantly on the server. No browser mic permissions, no iframe issues, works everywhere.</div>', unsafe_allow_html=True)

    # Groq API status badge
    has_key = bool(st.secrets.get("GROQ_API_KEY","") or os.environ.get("GROQ_API_KEY",""))
    bc = "#52b788" if has_key else "#ef4444"
    bt = "GROQ WHISPER READY" if has_key else "MISSING: ADD GROQ_API_KEY TO SECRETS"
    st.markdown(f"""
    <div class="groq-tag">
        <div class="groq-dot" style="background:{bc}"></div>
        <span style="color:{bc}">{bt}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="mic-panel">
        <div style="font-family:monospace;font-size:0.7rem;letter-spacing:2px;color:#52b788;margin-bottom:10px;">HOW TO USE VOICE INPUT</div>
        <ol>
            <li><strong>Phone:</strong> Open <em>Voice Memo</em> (iPhone) or <em>Recorder</em> (Android) → record your farm description → save/export as MP3 or M4A</li>
            <li><strong>PC:</strong> Open <em>Sound Recorder</em> (Windows) or <em>QuickTime</em> (Mac) → record → save as WAV or MP3</li>
            <li>Upload that file using the uploader below</li>
            <li>Click <strong>"Transcribe with Groq Whisper"</strong> — text appears in the box automatically</li>
            <li>Click <strong>"Analyze My Farm"</strong></li>
        </ol>
    </div>""", unsafe_allow_html=True)

    audio_file = st.file_uploader(
        "Upload voice recording (WAV · MP3 · MP4 · M4A · OGG · WEBM · FLAC)",
        type=["wav","mp3","mp4","m4a","ogg","webm","flac"],
        help="Record on your phone/PC and upload. Groq Whisper handles all formats.",
    )

    if audio_file is not None:
        st.audio(audio_file, format=audio_file.type)
        if st.button("🎙  Transcribe with Groq Whisper", key="transcribe_btn"):
            with st.spinner("Transcribing via Groq Whisper (usually < 3 seconds)..."):
                audio_bytes = audio_file.read()
                result = transcribe_with_groq(audio_bytes, filename=audio_file.name)
            if result:
                st.success(f"✅ Transcribed: **{result}**")
                st.session_state["voice_text"] = result
            else:
                st.warning("Could not transcribe. Speak clearly or type your description below.")

    # Examples
    st.markdown("**OR TRY AN EXAMPLE:**")
    examples = [
        "My soil is black, I get heavy monsoon rain, temperature is very hot",
        "Red sandy soil, less rainfall, moderate temperature, dry season",
        "Fertile dark soil near coast, very humid, lots of rain",
    ]
    ec = st.columns(3)
    for i, (col, phrase) in enumerate(zip(ec, examples)):
        with col:
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                st.session_state["voice_text"] = phrase

    voice_text = st.text_area(
        "Farm description:",
        value=st.session_state.get("voice_text",""),
        height=100,
        placeholder="E.g. 'Black fertile soil, heavy monsoon, very hot temperature near coastal area...'",
        key="voice_textarea",
    )

    if st.button("🌾  Analyze My Farm", key="voice_analyze"):
        if voice_text.strip():
            with st.spinner("Analyzing..."):
                features = parse_voice_input(voice_text)
                crop, conf, top3, shap_s, _ = predict_and_explain(features)
            with st.expander("📋 Values extracted from your description"):
                for feat, val in features.items():
                    unit = '°C' if feat=='temperature' else '%' if feat=='humidity' else ' mm' if feat=='rainfall' else ' kg/ha' if feat in ['N','P','K'] else ''
                    st.markdown(f"- {FEATURE_LABELS[feat]}: **{val:.1f}{unit}**")
            render_result(crop, conf, top3, shap_s, features)
        else:
            st.warning("Please upload a voice file or type a description first.")

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Manual Input
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Manual Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-sub">Enter precise soil and climate values. Sidebar sliders also update these fields.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        m_N  = st.number_input("Nitrogen (N) kg/ha",   0.0, 140.0, float(N),           step=1.0)
        m_P  = st.number_input("Phosphorus (P) kg/ha", 5.0, 145.0, float(P),           step=1.0)
        m_K  = st.number_input("Potassium (K) kg/ha",  5.0, 205.0, float(K),           step=1.0)
        m_ph = st.number_input("Soil pH",              3.5, 9.5,   float(ph),          step=0.1)
    with c2:
        m_temp = st.number_input("Temperature (°C)",   5.0, 45.0,  float(temperature), step=0.5)
        m_hum  = st.number_input("Humidity (%)",      20.0,100.0,  float(humidity),    step=1.0)
        m_rain = st.number_input("Rainfall (mm)",     20.0,300.0,  float(rainfall),    step=5.0)

    if st.button("🌾  Get Recommendation", key="manual_btn"):
        features = dict(N=m_N, P=m_P, K=m_K,
                        temperature=m_temp, humidity=m_hum,
                        ph=m_ph, rainfall=m_rain)
        crop, conf, top3, shap_s, _ = predict_and_explain(features)
        render_result(crop, conf, top3, shap_s, features)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="border-bottom:1px solid #2e2e2e;padding-bottom:24px;margin-bottom:32px;">
        <div style="font-family:monospace;font-size:0.7rem;letter-spacing:2px;color:#555;margin-bottom:8px;">HOW THE AI THINKS</div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:2.5rem;color:#fff;letter-spacing:2px;">MODEL & XAI INSIGHTS</div>
        <div style="color:#555;font-size:0.88rem;margin-top:6px;">Every prediction is explainable. No black boxes. This is the FarmVoice AI promise.</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2,1])
    with c1:
        fig2 = plot_importance()
        st.pyplot(fig2, use_container_width=True)
        plt.close()
    with c2:
        st.markdown("""
        <div style="background:#111;border:1px solid #2e2e2e;border-radius:4px;padding:24px;margin-bottom:16px;">
            <div style="font-family:monospace;font-size:0.65rem;letter-spacing:2px;color:#555;margin-bottom:12px;">WHAT IS XAI?</div>
            <div style="color:#fff;font-weight:500;margin-bottom:10px;">Explainable AI makes every decision transparent.</div>
            <div style="color:#888;font-size:0.85rem;line-height:1.8;">
                Instead of a black box saying <span style="color:#fff">"grow rice"</span> —
                FarmVoice AI tells you <span style="color:#52b788;font-weight:600">WHY</span>.<br><br>
                We use <span style="color:#fff">SHAP</span> (SHapley Additive Explanations) — a game-theory method
                that assigns each input an exact contribution per prediction.<br><br>
                Farmers must <em style="color:#52b788">trust</em> AI to act on it.
            </div>
        </div>
        <div style="background:#111;border:1px solid #2e2e2e;border-radius:4px;padding:20px;">
            <div style="font-family:monospace;font-size:0.65rem;letter-spacing:2px;color:#555;margin-bottom:14px;">MODEL ARCHITECTURE</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
                <tr><td style="color:#555;padding:7px 0;border-bottom:1px solid #1e1e1e;">Algorithm</td><td style="color:#fff;text-align:right;padding:7px 0;border-bottom:1px solid #1e1e1e;">Random Forest · 200 trees</td></tr>
                <tr><td style="color:#555;padding:7px 0;border-bottom:1px solid #1e1e1e;">Accuracy</td><td style="color:#52b788;font-weight:700;text-align:right;padding:7px 0;border-bottom:1px solid #1e1e1e;">89.1% on test data</td></tr>
                <tr><td style="color:#555;padding:7px 0;border-bottom:1px solid #1e1e1e;">Features</td><td style="color:#fff;text-align:right;padding:7px 0;border-bottom:1px solid #1e1e1e;">7 soil & climate inputs</td></tr>
                <tr><td style="color:#555;padding:7px 0;border-bottom:1px solid #1e1e1e;">Crops</td><td style="color:#fff;text-align:right;padding:7px 0;border-bottom:1px solid #1e1e1e;">22 Indian crops</td></tr>
                <tr><td style="color:#555;padding:7px 0;border-bottom:1px solid #1e1e1e;">XAI Method</td><td style="color:#fff;text-align:right;padding:7px 0;border-bottom:1px solid #1e1e1e;">SHAP TreeExplainer</td></tr>
                <tr><td style="color:#555;padding:7px 0;border-bottom:1px solid #1e1e1e;">Voice API</td><td style="color:#52b788;text-align:right;padding:7px 0;border-bottom:1px solid #1e1e1e;">Groq Whisper (free)</td></tr>
                <tr><td style="color:#555;padding:7px 0;">Deployment</td><td style="color:#fff;text-align:right;padding:7px 0;">Streamlit Cloud</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:32px;margin-bottom:16px;font-family:monospace;font-size:0.65rem;
    letter-spacing:2px;color:#555;text-transform:uppercase;">All 22 Supported Crops</div>
    """, unsafe_allow_html=True)
    crops_list = list(le.classes_)
    crop_cols  = st.columns(6)
    for i, c in enumerate(crops_list):
        with crop_cols[i % 6]:
            em = CROP_EMOJI.get(c,"🌱")
            st.markdown(
                f"<div style='background:#111;border:1px solid #222;border-radius:3px;"
                f"padding:10px 6px;text-align:center;font-size:0.82rem;color:#ccc;"
                f"margin:3px 0;font-family:monospace;'>{em}<br>{c.capitalize()}</div>",
                unsafe_allow_html=True)
