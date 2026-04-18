# 🌾 FarmVoice AI
### Explainable Crop Advisory System for Indian Farmers

[![Streamlit App](https://farmvoice-ai-wzrascjjkgbdbhccc8equa.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)
![XAI](https://img.shields.io/badge/XAI-SHAP-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-89.1%25-brightgreen)

---

## 🌟 The Problem

India has **140 million smallholder farmers**. Most don't know their soil's NPK values. They only know:
- *"My soil is black and fertile"*
- *"I get heavy rain for 4 months"*
- *"It gets very hot in summer"*

Existing AI tools require structured data inputs — creating a barrier for farmers who think in natural language, not numbers.

**More critically** — even when AI recommends a crop, farmers won't follow it unless they **understand why**.

---

## 💡 The Solution: FarmVoice AI

A voice/text-powered crop recommendation system that:

1. **Accepts natural language** — farmer describes their farm in plain English, Hindi, or Kannada transliteration
2. **Extracts features using NLP** — maps descriptions to soil/climate parameters
3. **Predicts the best crop** — using a Random Forest model (89.1% accuracy)
4. **Explains the decision in plain English** — using SHAP values
5. **Shows visual explanations** — SHAP waterfall charts any farmer can understand

> *This is exactly the philosophy Jacob-AI is built on — AI must be explainable and accessible to be trusted.*

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🎙️ Voice / Text Input | Describe your farm naturally |
| 🧠 NLP Parser | Converts natural language to farm parameters |
| 🌾 22 Crops | Rice, Wheat, Maize, Cotton, Fruits and more |
| 📊 SHAP Explainability | Every prediction is explained |
| 🗣️ Plain English Reasons | "Your high rainfall strongly favors Rice" |
| 🎛️ Manual Input | Precise slider/number inputs too |
| 📱 Deployed | Live on Streamlit Cloud |

---

## 📊 Model Performance

- **Algorithm:** Random Forest Classifier (200 estimators)
- **Accuracy:** 89.1% on held-out test set
- **Features:** N, P, K (soil nutrients), Temperature, Humidity, pH, Rainfall
- **Classes:** 22 Indian crops
- **Explainability:** SHAP TreeExplainer

---

## 🛠️ Tech Stack

```
Python 3.9+
├── Streamlit       → Web UI & deployment
├── Scikit-learn    → Random Forest model
├── SHAP            → Explainable AI
├── Pandas/NumPy    → Data processing
└── Matplotlib      → SHAP visualizations
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/AkashMs24/FarmVoice-AI.git
cd FarmVoice-AI

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → Select your repo → Set `app.py` as main file
4. Click **Deploy** — live in 2 minutes!

---

## 💬 Example Voice Inputs

```
"My soil is black and fertile, I get heavy monsoon rain, very hot temperature"
→ Recommends: Rice (with SHAP explanation)

"Red sandy soil, less rainfall, moderate temperature, hill area"  
→ Recommends: Wheat (with SHAP explanation)

"Coastal area, very humid, heavy rain, black soil"
→ Recommends: Coconut (with SHAP explanation)
```

---

## 🔍 Why Explainable AI Matters in Agriculture

Traditional AI: *"Grow Rice."* ❌ — Farmer doesn't trust it.

FarmVoice AI: *"Grow Rice because your high rainfall (250mm) strongly supports it, your temperature (30°C) is ideal, and your black soil pH (7.8) is suitable."* ✅ — Farmer understands and trusts it.

This is the core principle of **Explainable AI (XAI)** — making AI decisions transparent so humans can verify, trust, and act on them.

---

## 👨‍💻 Built By

**Akash M S** | Data Science & AI Student  
Presidency University, Bangalore  
📧 manigarakash@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/akashms01) | [GitHub](https://github.com/AkashMs24)

---

## 📁 Project Structure

```
FarmVoice-AI/
├── app.py                  # Main Streamlit application
├── model.pkl               # Trained Random Forest model
├── label_encoder.pkl       # Label encoder for crop names
├── crop_data.csv           # Training dataset (22 crops)
├── requirements.txt        # Python dependencies
├── train_model.py          # Model training script
└── README.md               # This file
```

---

## 🌐 Related Projects

- [Employee Attrition XAI](https://github.com/AkashMs24/Employee-Attrition-Risk-Assessment-Using-Explainable-Machine-Learning)
- [Fraud Detection System](https://github.com/AkashMs24/Cost-Sensitive-Real-Time-Fraud-Detection-Decision-System)
- [Explainable Recommendation Engine](https://github.com/AkashMs24/Domain-Agnostic-Explainable-Recommendation-Engine-for-Cold-Start-Scenarios)

---

*Built with 💚 for Indian farmers | Powered by XAI*
