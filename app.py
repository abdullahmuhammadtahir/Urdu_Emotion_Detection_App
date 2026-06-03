import streamlit as st
import torch
import numpy as np
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ✅ LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():

    model_path = "abd12-tahir/urdu-emotion-model"

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)

    device = "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()

# =========================
# ✅ LOAD LABEL ENCODER
# =========================
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# =========================
# ✅ EMOJI MAPPING
# =========================
emotion_emoji = {
    "happy": "😊",
    "sad": "😢",
    "fear": "😨",
    "anger": "😠",
    "love": "❤️",
    "neutral": "😐"
}

# =========================
# ✅ SINGLE SENTENCE FUNCTION
# =========================
def predict_final(text):

    text = text.strip()

    if not text:
        return {"emotion": "neutral", "confidence": 0.0}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    pred = le.classes_[np.argmax(probs)]
    conf = float(np.max(probs))

    # ✅ CONTRAST
    for word in ["لیکن", "مگر"]:
        if word in text:
            second = text.split(word)[-1].strip()
            if second:
                return predict_final(second)

    # ✅ NEGATION (FINAL FIXED ✅)
    if "نہیں" in text:
        if any(p in text for p in [
            "نہیں ہوں", "نہیں ہے", "نہیں لگ",
            "نہیں کرتا", "نہیں کرتی"
        ]):
            if pred in ["happy", "love"]:
                pred = "sad"
            elif pred in ["sad", "fear", "anger"]:
                pred = "neutral"

    return {"emotion": pred, "confidence": round(conf, 3)}

# =========================
# ✅ PARAGRAPH FUNCTION
# =========================
def predict_paragraph(text):

    text = text.strip()

    if not text:
        return {"emotion": "neutral", "details": []}

    text = text.replace("؟", ".").replace("!", ".").replace("۔", ".")

    sentences = [s.strip() for s in text.split(".") if s.strip()]

    results = []
    scores = {}

    for sentence in sentences:
        res = predict_final(sentence)

        results.append({
            "text": sentence,
            "emotion": res["emotion"],
            "confidence": res["confidence"]
        })

        scores[res["emotion"]] = scores.get(res["emotion"], 0) + res["confidence"]

    final = max(scores, key=scores.get) if scores else "neutral"

    return {"emotion": final, "details": results}

# =========================
# ✅ AUTO DETECTION
# =========================
def predict_auto(text):

    check = text.replace("؟", ".").replace("!", ".").replace("۔", ".")
    sentences = [s for s in check.split(".") if s.strip()]

    if len(sentences) == 1:
        return "sentence", predict_final(text)
    else:
        return "paragraph", predict_paragraph(text)

# =========================
# ✅ STREAMLIT UI
# =========================
st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("🧠 Urdu Emotion Detection (XLM-R + Hybrid)")
st.write("Enter Urdu text (single sentence or paragraph)")

user_input = st.text_area("Enter text here:", height=150)

if st.button("Predict Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        mode, result = predict_auto(user_input)

        st.subheader("✅ Prediction Result")

        # ✅ SENTENCE
        if mode == "sentence":
            emoji = emotion_emoji.get(result["emotion"], "")
            st.success(f"{emoji} Emotion: {result['emotion']}")
            st.write(f"Confidence: {result['confidence']}")

        # ✅ PARAGRAPH
        else:
            emoji = emotion_emoji.get(result["emotion"], "")
            st.success(f"{emoji} Final Emotion: {result['emotion']}")

            st.write("### Sentence-wise Breakdown:")

            emotions_list = []

            for r in result["details"]:
                emoji = emotion_emoji.get(r["emotion"], "")
                emotions_list.append(r["emotion"])

                st.write(f"👉 {r['text']}")
                st.write(f"{emoji} Emotion: {r['emotion']} | Confidence: {r['confidence']}")
                st.write("---")

            # ✅ BAR CHART
            df = pd.DataFrame(emotions_list, columns=["Emotion"])
            chart_data = df["Emotion"].value_counts()

            st.write("### 📊 Emotion Distribution")
            st.bar_chart(chart_data)

            # ✅ PIE CHART
            fig, ax = plt.subplots()
            chart_data.plot.pie(ax=ax, autopct='%1.1f%%')
            ax.set_ylabel("")

            st.write("### 🎯 Emotion Pie Chart")
            st.pyplot(fig)
