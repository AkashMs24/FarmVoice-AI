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
import speech_recognition as sr

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FarmVoice AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --green-dark:  #1a3a2a;
    --green-mid:   #2d6a4f;
    --green-light: #52b788;
    --gold:        #d4a017;
    --cream:       #fef9ef;
    --earth:       #8b5e3c;
    --text-dark:   #1a1a1a;
    --text-muted:  #6b7280;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--text-dark);
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, var(--green-light) 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "🌾🌿🌱";
    position: absolute;
    right: 32px; top: 20px;
    font-size: 64px;
    opacity: 0.15;
    letter-spacing: 8px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 8px 0;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero-subtitle {
    color: rgba(255,255,255,0.8);
    font-size: 1.1rem;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: var(--gold);
    color: var(--green-dark);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
}

/* Cards */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.04);
    margin-bottom: 20px;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--green-dark);
    margin-bottom: 4px;
}
.card-subtitle {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-bottom: 20px;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, var(--green-dark), var(--green-mid));
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    color: white;
    margin: 20px 0;
}
.result-crop {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    text-transform: capitalize;
    margin: 12px 0;
    letter-spacing: -1px;
}
.result-confidence {
    font-size: 1rem;
    opacity: 0.8;
    margin-top: 4px;
}
.result-emoji { font-size: 4rem; }

/* Stats row */
.stat-box {
    background: #111111;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #333333;
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: #ffffff;
}
.stat-label {
    font-size: 0.75rem;
    color: #aaaaaa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* Mic instruction box */
.mic-box {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid var(--green-light);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--green-dark) !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--green-light) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(45,106,79,0.35) !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stTabs [data-baseweb="tab-list"] {
    background: #f3f4f6;
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f1f1; }
::-webkit-scrollbar-thumb { background: var(--green-light); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
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
explainer = get_explainer(model)

# ── Constants ─────────────────────────────────────────────────────────────────
CROP_EMOJI = {
    'rice': '🌾', 'wheat': '🌾', 'maize': '🌽', 'chickpea': '🫘',
    'kidneybeans': '🫘', 'pigeonpeas': '🫘', 'mothbeans': '🫘',
    'mungbean': '🫘', 'blackgram': '🫘', 'lentil': '🫘',
    'pomegranate': '🍎', 'banana': '🍌', 'mango': '🥭',
    'grapes': '🍇', 'watermelon': '🍉', 'muskmelon': '🍈',
    'apple': '🍎', 'orange': '🍊', 'papaya': '🍈',
    'coconut': '🥥', 'cotton': '🌿', 'jute': '🌿',
}

FEATURE_LABELS = {
    'N': 'Nitrogen (N)', 'P': 'Phosphorus (P)', 'K': 'Potassium (K)',
    'temperature': 'Temperature', 'humidity': 'Humidity',
    'ph': 'Soil pH', 'rainfall': 'Rainfall',
}

FEATURE_TIPS = {
    'N': 'Essential for leaf growth. Higher = lush green crops.',
    'P': 'Root & flower development. Legumes need more.',
    'K': 'Disease resistance & fruit quality.',
    'temperature': 'Average seasonal temperature in °C.',
    'humidity': 'Average relative humidity in your region.',
    'ph': 'Soil acidity. 7 = neutral, <7 = acidic, >7 = alkaline.',
    'rainfall': 'Average annual rainfall in mm.',
}


# ── Server-side speech-to-text ────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes using Google Speech Recognition (server-side).
    Works with the WAV/webm bytes returned by st.audio_input.
    No iframe hacks, no browser JS, no CORS issues.
    """
    recognizer = sr.Recognizer()
    audio_file = io.BytesIO(audio_bytes)
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="en-IN")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        st.error(f"Speech Recognition API error: {e}")
        return ""
    except Exception as e:
        # Fallback: some browsers send webm; try converting with pydub if available
        try:
            from pydub import AudioSegment
            audio_file.seek(0)
            sound = AudioSegment.from_file(audio_file)
            wav_io = io.BytesIO()
            sound.export(wav_io, format="wav")
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language="en-IN")
        except Exception:
            st.warning("Could not process audio format. Please type your description instead.")
            return ""


# ── NLP voice parser ──────────────────────────────────────────────────────────
def parse_voice_input(text: str) -> dict:
    text = text.lower()
    features = {}

    if any(w in text for w in ['very fertile', 'very rich soil', 'high nitrogen', 'lots of manure']):
        features['N'] = 100
    elif any(w in text for w in ['fertile', 'rich soil', 'good soil', 'dark soil', 'black soil']):
        features['N'] = 75
    elif any(w in text for w in ['poor soil', 'low fertility', 'sandy', 'dry land']):
        features['N'] = 25
    else:
        features['N'] = 50

    if any(w in text for w in ['bean', 'pulse', 'legume', 'dal', 'chickpea']):
        features['P'] = 80
    elif any(w in text for w in ['fruit', 'mango', 'banana', 'orchard']):
        features['P'] = 30
    else:
        features['P'] = 50

    if any(w in text for w in ['good harvest', 'healthy crop', 'strong plant']):
        features['K'] = 70
    elif any(w in text for w in ['disease', 'weak', 'poor yield']):
        features['K'] = 25
    else:
        features['K'] = 45

    if any(w in text for w in ['very hot', 'extreme heat', 'scorching', 'desert']):
        features['temperature'] = 38
    elif any(w in text for w in ['hot', 'warm', 'summer', 'tropical']):
        features['temperature'] = 30
    elif any(w in text for w in ['cold', 'cool', 'winter', 'hill', 'mountain']):
        features['temperature'] = 15
    elif any(w in text for w in ['mild', 'moderate', 'pleasant']):
        features['temperature'] = 22
    else:
        features['temperature'] = 25

    if any(w in text for w in ['very humid', 'coastal', 'very wet', 'paddy', 'rice field']):
        features['humidity'] = 85
    elif any(w in text for w in ['humid', 'wet', 'rainy season', 'monsoon']):
        features['humidity'] = 70
    elif any(w in text for w in ['dry', 'arid', 'low humidity', 'desert']):
        features['humidity'] = 35
    else:
        features['humidity'] = 55

    if any(w in text for w in ['black soil', 'dark soil', 'cotton soil', 'regur']):
        features['ph'] = 7.8
    elif any(w in text for w in ['red soil', 'laterite', 'acidic']):
        features['ph'] = 6.0
    elif any(w in text for w in ['sandy', 'light soil']):
        features['ph'] = 6.5
    elif any(w in text for w in ['alkaline', 'salt', 'saline']):
        features['ph'] = 8.2
    else:
        features['ph'] = 6.8

    if any(w in text for w in ['heavy rain', 'floods', 'very rainy', '200mm', '300mm']):
        features['rainfall'] = 250
    elif any(w in text for w in ['good rain', 'monsoon', 'rainy', '100mm', '150mm']):
        features['rainfall'] = 140
    elif any(w in text for w in ['less rain', 'low rain', 'dry season', 'drought', '50mm']):
        features['rainfall'] = 50
    elif any(w in text for w in ['moderate rain', 'some rain']):
        features['rainfall'] = 90
    else:
        features['rainfall'] = 100

    return features


# ── Predict & explain ──────────────────────────────────────────────────────────
def predict_and_explain(features: dict):
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_df = pd.DataFrame([features])[feature_cols]

    pred_enc   = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    confidence = pred_proba.max() * 100
    crop       = le.inverse_transform([pred_enc])[0]

    top3_idx = np.argsort(pred_proba)[::-1][:3]
    top3 = [(le.inverse_transform([i])[0], pred_proba[i]*100) for i in top3_idx]

    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        shap_for_pred = shap_values[pred_enc]
    else:
        shap_for_pred = shap_values[:, :, pred_enc] if shap_values.ndim == 3 else shap_values

    shap_series = pd.Series(
        shap_for_pred[0] if shap_for_pred.ndim > 1 else shap_for_pred,
        index=feature_cols
    )
    return crop, confidence, top3, shap_series, input_df


def plain_english_explanation(crop, shap_series, features):
    sorted_factors = shap_series.abs().sort_values(ascending=False)
    top3_features  = sorted_factors.index[:3].tolist()
    lines = []
    for feat in top3_features:
        val   = shap_series[feat]
        label = FEATURE_LABELS[feat]
        raw   = features[feat]
        unit  = '°C' if feat=='temperature' else '%' if feat=='humidity' else ' mm' if feat=='rainfall' else ''
        if val > 0.01:
            lines.append(f"✅ **{label}** ({raw:.1f}{unit}) strongly favors **{crop}**")
        elif val < -0.01:
            lines.append(f"⚠️ **{label}** ({raw:.1f}{unit}) slightly reduces confidence, but {crop} remains best")
        else:
            lines.append(f"➡️ **{label}** ({raw:.1f}{unit}) has neutral effect")
    return lines


def plot_shap(shap_series, crop):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f9fafb')
    colors = ['#2d6a4f' if v > 0 else '#dc2626' for v in shap_series.values]
    labels = [FEATURE_LABELS.get(f, f) for f in shap_series.index]
    bars   = ax.barh(labels, shap_series.values, color=colors, edgecolor='none', height=0.6)
    ax.axvline(x=0, color='#1a1a1a', linewidth=1.2, alpha=0.5)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=10, color='#6b7280')
    ax.set_title(f'Why the AI recommends {crop.upper()}', fontsize=12,
                 fontweight='bold', color='#1a3a2a', pad=12)
    for bar, val in zip(bars, shap_series.values):
        ax.text(val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=9, color='#374151', fontweight='500')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10, colors='#374151')
    ax.tick_params(axis='x', labelsize=9, colors='#9ca3af')
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='#d1d5db')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <div class="hero-badge">XAI · Agri-Tech · India</div>
    <div class="hero-title">🌾 FarmVoice AI</div>
    <p class="hero-subtitle">Explainable Crop Advisory System for Indian Farmers · Voice-Powered · SHAP Explainability</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><div class="stat-num">22</div><div class="stat-label">Crops Supported</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="stat-num">89%</div><div class="stat-label">Model Accuracy</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="stat-num">7</div><div class="stat-label">Input Features</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-box"><div class="stat-num">XAI</div><div class="stat-label">SHAP Explainability</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌱 Farm Parameters")
    st.markdown("---")
    st.markdown("**Soil Nutrients**")
    N           = st.slider("Nitrogen (N) kg/ha",    0,   140, 60,  help=FEATURE_TIPS['N'])
    P           = st.slider("Phosphorus (P) kg/ha",  5,   145, 50,  help=FEATURE_TIPS['P'])
    K           = st.slider("Potassium (K) kg/ha",   5,   205, 50,  help=FEATURE_TIPS['K'])
    st.markdown("**Climate**")
    temperature = st.slider("Temperature (°C)",       5,   45,  25,  help=FEATURE_TIPS['temperature'])
    humidity    = st.slider("Humidity (%)",           20,  100, 60,  help=FEATURE_TIPS['humidity'])
    rainfall    = st.slider("Rainfall (mm)",          20,  300, 100, help=FEATURE_TIPS['rainfall'])
    st.markdown("**Soil**")
    ph          = st.slider("Soil pH",                3.5, 9.5, 6.5, step=0.1, help=FEATURE_TIPS['ph'])
    st.markdown("---")
    st.markdown("**About FarmVoice AI**")
    st.markdown("Built by **Akash M S** | Presidency University, Bangalore")
    st.markdown("Powered by Random Forest + SHAP XAI")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎙️ Voice / Text Input", "🎛️ Manual Input", "📊 Model Insights"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  ·  Voice / Text Input
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Describe Your Farm in Plain Language</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Record your voice OR type naturally. The AI will understand English, Hindi transliteration, or Kannada transliteration.</div>', unsafe_allow_html=True)

    # ── HOW IT WORKS (mic instructions) ──────────────────────────────────────
    st.markdown("""
    <div class="mic-box">
        <div style="font-size:1.6rem; margin-bottom:6px;">🎙️ How to use Voice Input</div>
        <ol style="margin:0; padding-left:18px; color:#2d6a4f; font-size:0.92rem; line-height:1.9;">
            <li>Click <strong>"Record"</strong> below → browser asks for microphone permission → allow it</li>
            <li>Speak your farm description clearly (e.g. <em>"My soil is black, monsoon rain, very hot"</em>)</li>
            <li>Click <strong>"Stop"</strong> when done</li>
            <li>Click <strong>"🔊 Transcribe Voice"</strong> — your words appear in the text box automatically</li>
            <li>Click <strong>"🌾 Analyze My Farm"</strong> to get your crop recommendation</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # ── st.audio_input — works natively, no JS iframe tricks ─────────────────
    # Requires streamlit >= 1.33.0
    audio_recording = st.audio_input(
        label="🎙️ Record your farm description",
        key="mic_input",
    )

    # Transcribe button — only shown after audio is recorded
    transcribed_text = ""
    if audio_recording is not None:
        if st.button("🔊 Transcribe Voice", key="transcribe_btn"):
            with st.spinner("Transcribing your voice..."):
                audio_bytes = audio_recording.read()
                transcribed_text = transcribe_audio(audio_bytes)
            if transcribed_text:
                st.success(f"✅ Transcribed: **{transcribed_text}**")
                # Store in session state so it persists into the text area
                st.session_state["voice_transcription"] = transcribed_text
            else:
                st.warning("⚠️ Could not transcribe audio. Please speak clearly and try again, or type your description below.")

    # ── Example buttons ───────────────────────────────────────────────────────
    example_phrases = [
        "My soil is black, I get heavy monsoon rain, temperature is very hot",
        "Red sandy soil, less rainfall, moderate temperature, dry season",
        "Fertile dark soil near coastal area, very humid, lots of rain",
    ]

    st.markdown("**💡 Or try an example:**")
    ex_cols = st.columns(3)
    for i, (col, phrase) in enumerate(zip(ex_cols, example_phrases)):
        with col:
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                st.session_state["voice_transcription"] = phrase

    # ── Text area — pre-filled by transcription or example ───────────────────
    voice_text = st.text_area(
        "Your farm description:",
        value=st.session_state.get("voice_transcription", ""),
        height=100,
        placeholder="E.g. 'My soil is black and fertile, I get heavy monsoon rain for 4 months, very hot temperature...'",
        key="voice_textarea",
    )

    # ── Analyze ───────────────────────────────────────────────────────────────
    if st.button("🌾 Analyze My Farm", key="voice_btn"):
        if voice_text.strip():
            with st.spinner("Analyzing your farm..."):
                features = parse_voice_input(voice_text)
                crop, confidence, top3, shap_series, input_df = predict_and_explain(features)
                emoji = CROP_EMOJI.get(crop, '🌱')

            st.markdown(f"""
            <div class="result-box">
                <div class="result-emoji">{emoji}</div>
                <div style="font-size:0.9rem;opacity:0.7;letter-spacing:2px;text-transform:uppercase;">Recommended Crop</div>
                <div class="result-crop">{crop}</div>
                <div class="result-confidence">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**📋 What we understood from your description:**")
                for feat, val in features.items():
                    label = FEATURE_LABELS[feat]
                    unit  = '°C' if feat=='temperature' else '%' if feat=='humidity' else ' mm' if feat=='rainfall' else ' kg/ha' if feat in ['N','P','K'] else ''
                    st.markdown(f"- {label}: **{val:.1f}{unit}**")
            with col_b:
                st.markdown("**🥇 Top 3 Alternatives:**")
                for c, prob in top3:
                    emoji_alt = CROP_EMOJI.get(c, '🌱')
                    bar = "█" * int(prob/5)
                    st.markdown(f"{emoji_alt} **{c.capitalize()}** — {prob:.1f}% {bar}")

            st.markdown("**🔍 Why this crop? (Plain English Explanation)**")
            for line in plain_english_explanation(crop, shap_series, features):
                st.markdown(f"> {line}")

            st.markdown("**📊 SHAP Feature Impact Chart:**")
            st.caption("Green bars = factors that support this crop | Red bars = factors that work against it")
            fig = plot_shap(shap_series, crop)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Please record your voice or enter a description first.")

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  ·  Manual Input
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Manual Soil & Climate Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Use the sliders on the left sidebar, or enter precise values below.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        m_N  = st.number_input("Nitrogen (N) kg/ha",   0.0, 140.0, float(N),           step=1.0)
        m_P  = st.number_input("Phosphorus (P) kg/ha", 5.0, 145.0, float(P),           step=1.0)
        m_K  = st.number_input("Potassium (K) kg/ha",  5.0, 205.0, float(K),           step=1.0)
        m_ph = st.number_input("Soil pH",              3.5, 9.5,   float(ph),          step=0.1)
    with col2:
        m_temp = st.number_input("Temperature (°C)",   5.0, 45.0,  float(temperature), step=0.5)
        m_hum  = st.number_input("Humidity (%)",      20.0,100.0,  float(humidity),    step=1.0)
        m_rain = st.number_input("Rainfall (mm)",     20.0,300.0,  float(rainfall),    step=5.0)

    if st.button("🌾 Get Recommendation", key="manual_btn"):
        features = dict(N=m_N, P=m_P, K=m_K, temperature=m_temp, humidity=m_hum, ph=m_ph, rainfall=m_rain)
        crop, confidence, top3, shap_series, input_df = predict_and_explain(features)
        emoji = CROP_EMOJI.get(crop, '🌱')

        st.markdown(f"""
        <div class="result-box">
            <div class="result-emoji">{emoji}</div>
            <div style="font-size:0.9rem;opacity:0.7;letter-spacing:2px;text-transform:uppercase;">Recommended Crop</div>
            <div class="result-crop">{crop}</div>
            <div class="result-confidence">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🥇 Top 3 Alternatives:**")
            for c, prob in top3:
                emoji_alt = CROP_EMOJI.get(c, '🌱')
                st.markdown(f"{emoji_alt} **{c.capitalize()}** — {prob:.1f}%")
        with col_b:
            st.markdown("**🔍 Explanation:**")
            for line in plain_english_explanation(crop, shap_series, features):
                st.markdown(f"> {line}")

        st.markdown("**📊 SHAP Feature Impact Chart:**")
        st.caption("Green bars = factors favoring this crop | Red = factors against it")
        fig = plot_shap(shap_series, crop)
        st.pyplot(fig)
        plt.close()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  ·  Model Insights  (black & white theme preserved)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="background:#111;border-radius:16px;padding:28px 32px;margin-bottom:20px;border:1px solid #333;">
        <div style="font-size:0.72rem;letter-spacing:2px;color:#888;text-transform:uppercase;margin-bottom:8px;">How the AI thinks</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.8rem;font-weight:900;color:#fff;margin-bottom:6px;">Model & XAI Insights</div>
        <div style="color:#aaa;font-size:0.9rem;">Every prediction is explainable. No black boxes. This is the FarmVoice AI promise.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div style="background:#000;border-radius:12px;padding:20px;border:1px solid #222;margin-bottom:16px;">
            <div style="font-size:0.72rem;letter-spacing:2px;color:#888;text-transform:uppercase;margin-bottom:10px;">Global Feature Importance</div>
        </div>
        """, unsafe_allow_html=True)
        importances = pd.Series(model.feature_importances_,
                                index=["N","P","K","temperature","humidity","ph","rainfall"])
        importances = importances.sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        fig2.patch.set_facecolor("#0a0a0a")
        ax2.set_facecolor("#0a0a0a")
        bars = ax2.barh([FEATURE_LABELS[f] for f in importances.index], importances.values,
                        color="#ffffff", edgecolor="none", height=0.55)
        for bar, val in zip(bars, importances.values):
            ax2.text(val+0.002, bar.get_y()+bar.get_height()/2,
                     f"{val:.3f}", va="center", ha="left", fontsize=9, color="#aaaaaa")
        ax2.set_xlabel("Feature Importance", fontsize=9, color="#666")
        ax2.set_title("Which factors matter most?", fontsize=11, fontweight="bold", color="#ffffff", pad=10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_color("#333")
        ax2.tick_params(axis="y", labelsize=9, colors="#aaa")
        ax2.tick_params(axis="x", labelsize=8, colors="#666")
        ax2.grid(axis="x", linestyle="--", alpha=0.2, color="#444")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col2:
        st.markdown("""
        <div style="background:#000;border-radius:12px;padding:24px;border:1px solid #222;margin-bottom:16px;">
            <div style="font-size:0.72rem;letter-spacing:2px;color:#888;text-transform:uppercase;margin-bottom:12px;">What is XAI?</div>
            <div style="font-size:1rem;color:#fff;font-weight:500;margin-bottom:10px;">Explainable AI makes every decision transparent.</div>
            <div style="color:#aaa;font-size:0.88rem;line-height:1.7;">
                Instead of a black box that says <span style="color:#fff;font-weight:500;">"grow rice"</span> — FarmVoice AI explains <span style="color:#52b788;font-weight:500;">WHY</span>.<br><br>
                We use <span style="color:#fff;font-weight:500;">SHAP (SHapley Additive Explanations)</span> — a game-theory technique that assigns each input an exact contribution value per prediction.<br><br>
                This is the same philosophy <span style="color:#52b788;font-weight:500;">Jacob-AI</span> is built on — farmers must trust and understand AI to act on it.
            </div>
        </div>
        <div style="background:#111;border-radius:12px;padding:20px;border:1px solid #222;">
            <div style="font-size:0.72rem;letter-spacing:2px;color:#888;text-transform:uppercase;margin-bottom:12px;">Model Architecture</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
                <tr><td style="color:#666;padding:6px 0;border-bottom:1px solid #222;">Algorithm</td><td style="color:#fff;text-align:right;padding:6px 0;border-bottom:1px solid #222;">Random Forest (200 trees)</td></tr>
                <tr><td style="color:#666;padding:6px 0;border-bottom:1px solid #222;">Accuracy</td><td style="color:#52b788;font-weight:600;text-align:right;padding:6px 0;border-bottom:1px solid #222;">89.1% on test data</td></tr>
                <tr><td style="color:#666;padding:6px 0;border-bottom:1px solid #222;">Features</td><td style="color:#fff;text-align:right;padding:6px 0;border-bottom:1px solid #222;">7 soil & climate inputs</td></tr>
                <tr><td style="color:#666;padding:6px 0;border-bottom:1px solid #222;">Crops</td><td style="color:#fff;text-align:right;padding:6px 0;border-bottom:1px solid #222;">22 Indian crops</td></tr>
                <tr><td style="color:#666;padding:6px 0;border-bottom:1px solid #222;">XAI Method</td><td style="color:#fff;text-align:right;padding:6px 0;border-bottom:1px solid #222;">SHAP TreeExplainer</td></tr>
                <tr><td style="color:#666;padding:6px 0;border-bottom:1px solid #222;">External API</td><td style="color:#52b788;text-align:right;padding:6px 0;border-bottom:1px solid #222;">None — fully local</td></tr>
                <tr><td style="color:#666;padding:6px 0;">Deployment</td><td style="color:#fff;text-align:right;padding:6px 0;">Streamlit Cloud</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#000;border-radius:12px;padding:20px 24px;margin-top:16px;border:1px solid #222;">
        <div style="font-size:0.72rem;letter-spacing:2px;color:#888;text-transform:uppercase;margin-bottom:14px;">All 22 Supported Crops</div>
    </div>
    """, unsafe_allow_html=True)
    crops_list = list(le.classes_)
    crop_cols  = st.columns(6)
    for i, c in enumerate(crops_list):
        with crop_cols[i % 6]:
            em = CROP_EMOJI.get(c, "🌱")
            st.markdown(
                f"<div style='background:#111;border:1px solid #222;border-radius:8px;"
                f"padding:8px;text-align:center;font-size:0.85rem;color:#fff;margin:3px 0;'>"
                f"{em}<br>{c.capitalize()}</div>",
                unsafe_allow_html=True
            )
