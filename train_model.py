"""
FarmVoice AI — Model Training Script
Run this to retrain the model from scratch.
Author: Akash M S | Presidency University, Bangalore
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

def generate_dataset():
    """Generate realistic crop dataset for 22 Indian crops."""
    np.random.seed(42)

    crops = {
        'rice':        dict(N=(60,100), P=(30,60),  K=(30,60),  temp=(20,30), hum=(70,90), ph=(5.5,7.0), rain=(150,300)),
        'wheat':       dict(N=(80,120), P=(30,60),  K=(30,60),  temp=(10,25), hum=(40,70), ph=(6.0,7.5), rain=(50,150)),
        'maize':       dict(N=(60,100), P=(50,80),  K=(50,80),  temp=(18,30), hum=(50,75), ph=(5.5,7.5), rain=(60,150)),
        'chickpea':    dict(N=(10,40),  P=(60,100), K=(60,100), temp=(15,25), hum=(30,60), ph=(6.0,8.0), rain=(30,100)),
        'kidneybeans': dict(N=(10,40),  P=(60,100), K=(60,100), temp=(15,25), hum=(50,80), ph=(5.5,7.0), rain=(80,180)),
        'pigeonpeas':  dict(N=(10,40),  P=(40,70),  K=(40,70),  temp=(25,35), hum=(40,70), ph=(5.5,7.5), rain=(60,150)),
        'mothbeans':   dict(N=(10,40),  P=(40,70),  K=(40,70),  temp=(25,40), hum=(30,60), ph=(6.0,8.0), rain=(30,80)),
        'mungbean':    dict(N=(10,40),  P=(40,70),  K=(40,70),  temp=(25,35), hum=(60,80), ph=(6.0,7.5), rain=(60,120)),
        'blackgram':   dict(N=(10,40),  P=(40,70),  K=(40,70),  temp=(25,35), hum=(60,80), ph=(5.5,7.0), rain=(60,120)),
        'lentil':      dict(N=(10,40),  P=(60,100), K=(60,100), temp=(15,25), hum=(40,70), ph=(6.0,8.0), rain=(30,100)),
        'pomegranate': dict(N=(10,40),  P=(10,40),  K=(30,60),  temp=(25,35), hum=(40,70), ph=(5.5,7.5), rain=(50,150)),
        'banana':      dict(N=(80,120), P=(60,90),  K=(40,70),  temp=(25,35), hum=(70,90), ph=(5.5,7.0), rain=(100,200)),
        'mango':       dict(N=(10,40),  P=(10,40),  K=(10,40),  temp=(25,40), hum=(50,80), ph=(5.5,7.5), rain=(80,200)),
        'grapes':      dict(N=(10,40),  P=(10,40),  K=(30,60),  temp=(15,30), hum=(50,80), ph=(5.5,7.5), rain=(60,150)),
        'watermelon':  dict(N=(80,120), P=(10,40),  K=(40,70),  temp=(25,40), hum=(60,80), ph=(5.5,7.0), rain=(40,100)),
        'muskmelon':   dict(N=(80,120), P=(10,40),  K=(40,70),  temp=(28,38), hum=(50,75), ph=(6.0,7.5), rain=(30,80)),
        'apple':       dict(N=(0,20),   P=(100,140),K=(130,200),temp=(0,20),  hum=(50,80), ph=(5.5,7.0), rain=(100,200)),
        'orange':      dict(N=(0,20),   P=(5,20),   K=(5,20),   temp=(15,30), hum=(70,90), ph=(5.5,7.5), rain=(100,200)),
        'papaya':      dict(N=(40,60),  P=(10,60),  K=(40,60),  temp=(25,40), hum=(70,90), ph=(6.0,7.5), rain=(100,200)),
        'coconut':     dict(N=(0,20),   P=(0,20),   K=(30,60),  temp=(25,35), hum=(70,90), ph=(5.5,7.0), rain=(150,300)),
        'cotton':      dict(N=(100,140),P=(20,50),  K=(15,40),  temp=(24,35), hum=(50,75), ph=(6.0,8.0), rain=(60,150)),
        'jute':        dict(N=(60,90),  P=(40,70),  K=(40,70),  temp=(25,35), hum=(70,90), ph=(6.0,7.5), rain=(150,250)),
    }

    rows = []
    per_crop = 2200 // len(crops)
    for crop, ranges in crops.items():
        for _ in range(per_crop):
            rows.append({
                'N':           np.random.uniform(*ranges['N']),
                'P':           np.random.uniform(*ranges['P']),
                'K':           np.random.uniform(*ranges['K']),
                'temperature': np.random.uniform(*ranges['temp']),
                'humidity':    np.random.uniform(*ranges['hum']),
                'ph':          np.random.uniform(*ranges['ph']),
                'rainfall':    np.random.uniform(*ranges['rain']),
                'label':       crop,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def train():
    print("🌾 FarmVoice AI — Model Training")
    print("=" * 40)

    # Dataset
    df = generate_dataset()
    df.to_csv('crop_data.csv', index=False)
    print(f"✅ Dataset: {len(df)} rows, {df.label.nunique()} crops")

    X = df.drop('label', axis=1)
    y = df['label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    # Train
    print("\n🤖 Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Test Accuracy: {acc*100:.1f}%")

    cv_scores = cross_val_score(model, X, y_enc, cv=5)
    print(f"✅ Cross-Val Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))

    # Save
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("\n✅ Model saved: model.pkl")
    print("✅ Encoder saved: label_encoder.pkl")
    print("\n🚀 Run: streamlit run app.py")


if __name__ == '__main__':
    train()
