import os
import sys
import numpy as np
import torch
import torch.nn as nn
import librosa

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "audio_sentiment_v2.pth"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 130

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🎧 Using device:", DEVICE)

# --------------------------------------------------
# LABEL MAP (same as training)
# --------------------------------------------------
LABEL_MAP = {
    0: "angry",
    1: "fearful",
    2: "happy",
    3: "neutral",
    4: "sad",
    5: "surprise"
}
# --------------------------------------------------
# FEATURE EXTRACTION (same as training)
# --------------------------------------------------
def extract_mfcc(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # pad / trim
    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc

# --------------------------------------------------
# AUDIO CNN (MUST MATCH train_audio_v2.py)
# --------------------------------------------------
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    sys.exit(1)

model = AudioCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully:", MODEL_PATH)

# --------------------------------------------------
# INPUT AUDIO FILE
# --------------------------------------------------
if len(sys.argv) < 2:
    print("⚠️ Usage: python predict_audio.py <audio_file.wav>")
    sys.exit(0)

audio_path = sys.argv[1]

if not os.path.exists(audio_path):
    print("❌ Audio file not found:", audio_path)
    sys.exit(1)

# --------------------------------------------------
# PREDICT
# --------------------------------------------------
mfcc = extract_mfcc(audio_path)
X = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,40,130)
X = X.to(DEVICE)

with torch.no_grad():
    logits = model(X)
    pred = torch.argmax(logits, dim=1).item()

print("🎯 Predicted Emotion:", LABEL_MAP[pred])
