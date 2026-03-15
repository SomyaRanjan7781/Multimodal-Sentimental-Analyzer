import streamlit as st
import torch
import numpy as np
import tempfile
from PIL import Image
import librosa
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🔥 Multimodal Emotion AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:800;
    text-align:center;
}
.result-box {
    padding:15px;
    border-radius:10px;
    background-color:#1f1f1f;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'> Multimodal Sentiment Analyzer</div>", unsafe_allow_html=True)
st.write("Upload Image / Audio / Text and get Emotion Predictions")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LABELS ----------------
IMAGE_CLASSES = ["angry","disgust","fear","happy","neutral","sad","surprise"]
AUDIO_CLASSES = ["angry","fearful","happy","neutral","sad","surprise"]

# ---------------- IMAGE MODEL ----------------
@st.cache_resource
def load_image_model():
    from torchvision.models import resnet18
    import torch.nn as nn

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load("image_sentiment.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

image_model = load_image_model()

# ---------------- AUDIO MODEL ----------------
@st.cache_resource
def load_audio_model():
    from train_audio_v2 import AudioCNN
    model = AudioCNN()
    model.load_state_dict(torch.load("audio_sentiment_v2.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

audio_model = load_audio_model()

# ---------------- TEXT MODEL ----------------
@st.cache_resource
def load_text_model():
    import joblib
    return joblib.load("text_sentiment_llm.pkl")

text_model = load_text_model()

# ---------------- IMAGE TRANSFORM ----------------
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# ---------------- IMAGE PREDICT FUNCTION ----------------
def predict_image(img):
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = image_model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs,1)
    return IMAGE_CLASSES[pred.item()], conf.item()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# ================= IMAGE =================
with col1:
    st.subheader("🖼 Image Emotion")
    image_file = st.file_uploader("Upload Image", type=["jpg","png"])

    if image_file:
        img = Image.open(image_file).convert("RGB")
        st.image(img, width=300)

        emotion, confidence = predict_image(img)
        st.success(f"Predicted Emotion: {emotion}")
        st.write(f"Confidence: {confidence:.4f}")

# ================= WEBCAM =================
with col1:
    st.subheader("📷 Live Webcam Emotion")
    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        img = Image.open(camera_image).convert("RGB")
        st.image(img, width=300)

        emotion, confidence = predict_image(img)
        st.success(f"Predicted Emotion: {emotion}")
        st.write(f"Confidence: {confidence:.4f}")

# ================= AUDIO =================
with col2:
    st.subheader("🎧 Audio Emotion")
    audio_file = st.file_uploader("Upload Audio (.wav / .mp3)", type=["wav","mp3"])

    if audio_file:
        st.audio(audio_file)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        if mfcc.shape[1] < 130:
            pad = 130 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)))
        else:
            mfcc = mfcc[:,:130]

        tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            output = audio_model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs,1)

        st.success(f"Predicted Emotion: {AUDIO_CLASSES[pred.item()]}")
        st.write(f"Confidence: {conf.item():.4f}")

# ================= TEXT =================
st.subheader("📝 Text Sentiment")
text_input = st.text_area("Enter text")

if st.button("Predict Text"):
    if text_input.strip() != "":
        clf = text_model["classifier"]

        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        emb = embedder.encode([text_input])
        pred = clf.predict(emb)[0]
        proba = clf.predict_proba(emb)[0]

        label = "positive" if pred==1 else "negative"
        confidence = max(proba)

        st.success(f"Predicted Sentiment: {label}")
        st.write(f"Confidence: {confidence:.4f}")