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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FarmVoice AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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
.result-emoji {
    font-size: 4rem;
}

/* Explanation pill */
.factor-pill {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 4px;
}
.factor-positive { background: #d1fae5; color: #065f46; }
.factor-negative { background: #fee2e2; color: #991b1b; }

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

/* Voice box */
.voice-box {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px dashed var(--green-light);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--green-dark) !important;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
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

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #f3f4f6;
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
}

/* Scrollbar */
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

# ── Crop emoji map ────────────────────────────────────────────────────────────
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
    'N': 'Nitrogen (N)',
    'P': 'Phosphorus (P)',
    'K': 'Potassium (K)',
    'temperature': 'Temperature',
    'humidity': 'Humidity',
    'ph': 'Soil pH',
    'rainfall': 'Rainfall',
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

# ── NLP voice parser ──────────────────────────────────────────────────────────
def parse_voice_input(text: str) -> dict:
    """Extract farm features from natural language description."""
    text = text.lower()
    features = {}

    # Nitrogen
    if any(w in text for w in ['very fertile', 'very rich soil', 'high nitrogen', 'lots of manure']):
        features['N'] = 100
    elif any(w in text for w in ['fertile', 'rich soil', 'good soil', 'dark soil', 'black soil']):
        features['N'] = 75
    elif any(w in text for w in ['poor soil', 'low fertility', 'sandy', 'dry land']):
        features['N'] = 25
    else:
        features['N'] = 50

    # Phosphorus
    if any(w in text for w in ['bean', 'pulse', 'legume', 'dal', 'chickpea']):
        features['P'] = 80
    elif any(w in text for w in ['fruit', 'mango', 'banana', 'orchard']):
        features['P'] = 30
    else:
        features['P'] = 50

    # Potassium
    if any(w in text for w in ['good harvest', 'healthy crop', 'strong plant']):
        features['K'] = 70
    elif any(w in text for w in ['disease', 'weak', 'poor yield']):
        features['K'] = 25
    else:
        features['K'] = 45

    # Temperature
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

    # Humidity
    if any(w in text for w in ['very humid', 'coastal', 'very wet', 'paddy', 'rice field']):
        features['humidity'] = 85
    elif any(w in text for w in ['humid', 'wet', 'rainy season', 'monsoon']):
        features['humidity'] = 70
    elif any(w in text for w in ['dry', 'arid', 'low humidity', 'desert']):
        features['humidity'] = 35
    else:
        features['humidity'] = 55

    # pH
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

    # Rainfall
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

# ── Predict & explain ─────────────────────────────────────────────────────────
def predict_and_explain(features: dict):
    input_df = pd.DataFrame([features])
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_df = input_df[feature_cols]

    pred_enc = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    confidence = pred_proba.max() * 100
    crop = le.inverse_transform([pred_enc])[0]

    # Top 3 alternatives
    top3_idx = np.argsort(pred_proba)[::-1][:3]
    top3 = [(le.inverse_transform([i])[0], pred_proba[i]*100) for i in top3_idx]

    # SHAP
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
    """Generate human-readable explanation."""
    sorted_factors = shap_series.abs().sort_values(ascending=False)
    top3_features = sorted_factors.index[:3].tolist()

    lines = []
    for feat in top3_features:
        val = shap_series[feat]
        label = FEATURE_LABELS[feat]
        raw = features[feat]
        unit = '°C' if feat == 'temperature' else '%' if feat == 'humidity' else ' mm' if feat == 'rainfall' else ''
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

    bars = ax.barh(labels, shap_series.values, color=colors, edgecolor='none', height=0.6)

    ax.axvline(x=0, color='#1a1a1a', linewidth=1.2, alpha=0.5)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=10, color='#6b7280')
    ax.set_title(f'Why the AI recommends {crop.upper()}', fontsize=12, fontweight='bold', color='#1a3a2a', pad=12)

    for bar, val in zip(bars, shap_series.values):
        ax.text(
            val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height()/2,
            f'{val:+.3f}',
            va='center', ha='left' if val >= 0 else 'right',
            fontsize=9, color='#374151', fontweight='500'
        )

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

# Hero header
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">XAI · Agri-Tech · India</div>
    <div class="hero-title">🌾 FarmVoice AI</div>
    <p class="hero-subtitle">Explainable Crop Advisory System for Indian Farmers · Voice-Powered · SHAP Explainability</p>
</div>
""", unsafe_allow_html=True)

# Stats row
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌱 Farm Parameters")
    st.markdown("---")
    st.markdown("**Soil Nutrients**")
    N   = st.slider("Nitrogen (N) kg/ha",    0, 140, 60, help=FEATURE_TIPS['N'])
    P   = st.slider("Phosphorus (P) kg/ha",  5, 145, 50, help=FEATURE_TIPS['P'])
    K   = st.slider("Potassium (K) kg/ha",   5, 205, 50, help=FEATURE_TIPS['K'])
    st.markdown("**Climate**")
    temperature = st.slider("Temperature (°C)", 5, 45, 25, help=FEATURE_TIPS['temperature'])
    humidity    = st.slider("Humidity (%)",     20, 100, 60, help=FEATURE_TIPS['humidity'])
    rainfall    = st.slider("Rainfall (mm)",    20, 300, 100, help=FEATURE_TIPS['rainfall'])
    st.markdown("**Soil**")
    ph = st.slider("Soil pH", 3.5, 9.5, 6.5, step=0.1, help=FEATURE_TIPS['ph'])

    st.markdown("---")
    st.markdown("**About FarmVoice AI**")
    st.markdown("Built by **Akash M S** | Presidency University, Bangalore")
    st.markdown("Powered by Random Forest + SHAP XAI")
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import speech_recognition as sr
import tempfile
import wave

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

def speech_to_text(audio_frames):
    if len(audio_frames) == 0:
        return ""

    # Save audio temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    wf = wave.open(temp_file.name, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)

    audio_data = np.concatenate(audio_frames)
    wf.writeframes(audio_data.tobytes())
    wf.close()

    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_file.name) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
    except:
        text = "Could not understand audio"

    return text
# ── Main content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎙️ Voice / Text Input", "🎛️ Manual Input", "📊 Model Insights"])

# ─── TAB 1: Voice / Text ─────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Describe Your Farm in Plain Language</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Type naturally — in English, Hindi transliteration, or Kannada transliteration. The AI will understand.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="voice-box">
        <div style="font-size:2.5rem">🎙️</div>
        <div style="font-weight:600; color:#2d6a4f; margin:8px 0">Speak or Type Your Farm Description</div>
        <div style="font-size:0.85rem; color:#6b7280">Click below → allow mic → speak → your words appear instantly</div>
    </div>

    <div style="display:flex; justify-content:center; margin:14px 0; gap:12px; flex-wrap:wrap;">
        <button id="micBtn" onclick="startVoice()" style="background:#1a3a2a;color:white;border:none;padding:12px 32px;border-radius:24px;font-size:1rem;cursor:pointer;font-weight:600;">🎙️ Click to Speak</button>
        <button onclick="clearVoice()" style="background:transparent;color:#6b7280;border:1px solid #d1d5db;padding:12px 20px;border-radius:24px;font-size:0.9rem;cursor:pointer;">✕ Clear</button>
    </div>
    <div id="voiceStatus" style="text-align:center;font-size:0.85rem;color:#6b7280;margin:6px 0;min-height:22px;"></div>
    <div id="voiceResult" style="display:none;background:#111;border:2px solid #52b788;border-radius:10px;padding:14px 18px;font-size:1rem;color:#fff;margin:10px 0;line-height:1.5;">
        <span style="font-size:0.72rem;color:#52b788;letter-spacing:1px;text-transform:uppercase;">You said:</span><br>
        <span id="voiceText" style="font-weight:500;"></span>
    </div>
    <script>
    function startVoice() {
        var btn=document.getElementById('micBtn');
        var status=document.getElementById('voiceStatus');
        var resultDiv=document.getElementById('voiceResult');
        var voiceTextEl=document.getElementById('voiceText');
        var SR=window.SpeechRecognition||window.webkitSpeechRecognition;
        if(!SR){status.innerHTML='<span style="color:#dc2626">⚠️ Use Chrome or Edge for voice input</span>';return;}
        var r=new SR();
        r.lang='en-IN';r.interimResults=true;r.maxAlternatives=1;r.continuous=false;
        btn.innerHTML='🔴 Listening...';btn.style.background='#dc2626';
        status.innerHTML='<span style="color:#2d6a4f;font-weight:500;">🎤 Listening — describe your farm now</span>';
        r.start();
        r.onresult=function(e){
            var interim='',final='';
            for(var i=e.resultIndex;i<e.results.length;i++){
                if(e.results[i].isFinal){final+=e.results[i][0].transcript;}
                else{interim+=e.results[i][0].transcript;}
            }
            var cur=final||interim;
            voiceTextEl.innerText=cur;
            resultDiv.style.display='block';
            if(final){
                status.innerHTML='<span style="color:#2d6a4f">✅ Done! Copy the text above into the box below, then click Analyze</span>';
                try{
                    var tas=window.parent.document.querySelectorAll('textarea');
                    tas.forEach(function(ta){
                        var s=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;
                        s.call(ta,final);
                        ta.dispatchEvent(new Event('input',{bubbles:true}));
                    });
                }catch(err){}
            }
        };
        r.onerror=function(e){
            btn.innerHTML='🎙️ Click to Speak';btn.style.background='#1a3a2a';
            var msg=e.error==='not-allowed'?'❌ Mic blocked — allow mic in browser':'❌ '+e.error+' — try again';
            status.innerHTML='<span style="color:#dc2626">'+msg+'</span>';
        };
        r.onend=function(){btn.innerHTML='🎙️ Speak Again';btn.style.background='#1a3a2a';};
    }
    function clearVoice(){
        document.getElementById('voiceResult').style.display='none';
        document.getElementById('voiceText').innerText='';
        document.getElementById('voiceStatus').innerText='';
    }
    </script>
    """, unsafe_allow_html=True)

    example_phrases = [
        "My soil is black, I get heavy monsoon rain, temperature is very hot",
        "Red sandy soil, less rainfall, moderate temperature, dry season",
        "Fertile dark soil near coastal area, very humid, lots of rain",
        "Hill area, cold weather, moderate rain, acidic red soil",
        "Cotton growing region, hot and dry, black regur soil",
    ]

    st.markdown("**💡 Try an example:**")
    ex_cols = st.columns(len(example_phrases[:3]))
    selected_example = ""
    for i, (col, phrase) in enumerate(zip(ex_cols, example_phrases[:3])):
        with col:
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                selected_example = phrase

    voice_text = st.text_area(
        "Your farm description:",
        value=selected_example,
        height=100,
        placeholder="E.g. 'My soil is black and fertile, I get heavy monsoon rain for 4 months, very hot temperature...'",
    )

    if st.button("🌾 Analyze My Farm", key="voice_btn"):
        if voice_text.strip():
            with st.spinner("Analyzing your farm..."):
                features = parse_voice_input(voice_text)
                crop, confidence, top3, shap_series, input_df = predict_and_explain(features)
                emoji = CROP_EMOJI.get(crop, '🌱')

            # Result
            st.markdown(f"""
            <div class="result-box">
                <div class="result-emoji">{emoji}</div>
                <div style="font-size:0.9rem; opacity:0.7; letter-spacing:2px; text-transform:uppercase;">Recommended Crop</div>
                <div class="result-crop">{crop}</div>
                <div class="result-confidence">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown("**📋 What we understood from your description:**")
                for feat, val in features.items():
                    label = FEATURE_LABELS[feat]
                    unit = '°C' if feat == 'temperature' else '%' if feat == 'humidity' else ' mm' if feat == 'rainfall' else ' kg/ha' if feat in ['N','P','K'] else ''
                    st.markdown(f"- {label}: **{val:.1f}{unit}**")

            with col_b:
                st.markdown("**🥇 Top 3 Alternatives:**")
                for c, prob in top3:
                    emoji_alt = CROP_EMOJI.get(c, '🌱')
                    bar = "█" * int(prob/5)
                    st.markdown(f"{emoji_alt} **{c.capitalize()}** — {prob:.1f}% {bar}")

            st.markdown("**🔍 Why this crop? (Plain English Explanation)**")
            explanations = plain_english_explanation(crop, shap_series, features)
            for line in explanations:
                st.markdown(f"> {line}")

            st.markdown("**📊 SHAP Feature Impact Chart:**")
            st.caption("Green bars = factors that support this crop | Red bars = factors that work against it")
            fig = plot_shap(shap_series, crop)
            st.pyplot(fig)
            plt.close()

        else:
            st.warning("Please enter a description of your farm first.")

    st.markdown('</div>', unsafe_allow_html=True)

# ─── TAB 2: Manual Input ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Manual Soil & Climate Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Use the sliders on the left sidebar, or enter precise values below.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        m_N    = st.number_input("Nitrogen (N) kg/ha",    0.0, 140.0, float(N),   step=1.0)
        m_P    = st.number_input("Phosphorus (P) kg/ha",  5.0, 145.0, float(P),   step=1.0)
        m_K    = st.number_input("Potassium (K) kg/ha",   5.0, 205.0, float(K),   step=1.0)
        m_ph   = st.number_input("Soil pH",               3.5, 9.5,   float(ph),  step=0.1)
    with col2:
        m_temp = st.number_input("Temperature (°C)",      5.0, 45.0,  float(temperature), step=0.5)
        m_hum  = st.number_input("Humidity (%)",          20.0,100.0, float(humidity),    step=1.0)
        m_rain = st.number_input("Rainfall (mm)",         20.0,300.0, float(rainfall),    step=5.0)

    if st.button("🌾 Get Recommendation", key="manual_btn"):
        features = dict(N=m_N, P=m_P, K=m_K, temperature=m_temp, humidity=m_hum, ph=m_ph, rainfall=m_rain)
        crop, confidence, top3, shap_series, input_df = predict_and_explain(features)
        emoji = CROP_EMOJI.get(crop, '🌱')

        st.markdown(f"""
        <div class="result-box">
            <div class="result-emoji">{emoji}</div>
            <div style="font-size:0.9rem; opacity:0.7; letter-spacing:2px; text-transform:uppercase;">Recommended Crop</div>
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
            explanations = plain_english_explanation(crop, shap_series, features)
            for line in explanations:
                st.markdown(f"> {line}")

        st.markdown("**📊 SHAP Feature Impact Chart:**")
        st.caption("Green bars = factors favoring this crop | Red = factors against it")
        fig = plot_shap(shap_series, crop)
        st.pyplot(fig)
        plt.close()

    st.markdown('</div>', unsafe_allow_html=True)

# ─── TAB 3: Model Insights ──────────────────────────────────────────────────
with tab3:
    # Full black/white header
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
    crop_cols = st.columns(6)
    for i, c in enumerate(crops_list):
        with crop_cols[i % 6]:
            em = CROP_EMOJI.get(c, "🌱")
            st.markdown(f"<div style='background:#111;border:1px solid #222;border-radius:8px;padding:8px;text-align:center;font-size:0.85rem;color:#fff;margin:3px 0;'>{em}<br>{c.capitalize()}</div>", unsafe_allow_html=True)
