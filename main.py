from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import joblib
import os
import io
from pydub import AudioSegment

app = FastAPI()

# Load Models
model_drunk = joblib.load('model_drunk.pkl')
le_drunk = joblib.load('label_encoder_drunk.pkl')

def extract_features(y, sr):
    y, _ = librosa.effects.trim(y)
    if len(y) < sr:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1).tolist()
    mfcc_std = np.std(mfcc, axis=1).tolist()
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    tempo = float(librosa.beat.beat_track(y=y, sr=sr)[0])

    features = mfcc_mean + mfcc_std + [zcr, rms, centroid, tempo]
    return np.array(features).reshape(1, -1)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    
    # Load audio with librosa
    y, sr = librosa.load(audio_buffer, sr=None)

    # Extract features
    features = extract_features(y, sr)

    if features is None:
        return JSONResponse(content={"detail": "Audio too short for feature extraction."}, status_code=400)

    drunk_probs = model_drunk.predict_proba(features)[0]
    drunk_idx = np.argmax(drunk_probs)
    drunk_label = le_drunk.inverse_transform([drunk_idx])[0]
    drunk_conf = drunk_probs[drunk_idx] * 100

    return JSONResponse({"condition": drunk_label, "confidence": drunk_conf})
