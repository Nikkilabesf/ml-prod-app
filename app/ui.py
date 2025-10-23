import os
from pathlib import Path
import gradio as gr
import joblib
import pandas as pd

# --- Smart base directory detection (works everywhere) ---
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent.parent
else:
    BASE_DIR = Path(os.getcwd())

# --- Safe paths for artifacts ---
MODEL_PATH = BASE_DIR / "app" / "artifacts" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "app" / "artifacts" / "preprocessor.pkl"

print(f"📂 Loading model from: {MODEL_PATH}")
print(f"📂 Loading preprocessor from: {PREPROCESSOR_PATH}")

# --- Load model and preprocessor ---
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
print("✅ Model and preprocessor loaded successfully!")

# --- Prediction logic ---
def predict_risk(age, bmi, avg_glucose, sex, smoker, region):
    try:
        data = pd.DataFrame([{
            "age": float(age),
            "bmi": float(bmi),
            "avg_glucose": float(avg_glucose),
            "sex": sex.lower(),
            "smoker": smoker.lower(),
            "region": region.lower()
        }])
        X_proc = preprocessor.transform(data)
        pred = model.predict(X_proc)[0]
        proba = model.predict_proba(X_proc)[0][1]

        if pred == 1:
            msg = f"🚑 **High Risk** — Probability {proba:.1%}. You should visit the **ER immediately.**"
        else:
            msg = f"🩺 **Moderate/Low Risk** — Probability {proba:.1%}. You can schedule a **regular appointment.**"
        return msg
    except Exception as e:
        return f"⚠️ Error: {e}"

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple")) as app:
    gr.Markdown("## 💖 Diabetes Risk Chatbot\nAnswer a few questions to check your emergency risk level.")
    with gr.Row():
        age = gr.Number(label="Age", value=40)
        bmi = gr.Number(label="BMI", value=25.0)
        avg_glucose = gr.Number(label="Average Glucose", value=110)
    with gr.Row():
        sex = gr.Dropdown(["male", "female"], label="Sex", value="female")
        smoker = gr.Dropdown(["yes", "no"], label="Smoker", value="no")
        region = gr.Dropdown(["northeast", "northwest", "southeast", "southwest"], label="Region", value="northwest")

    output = gr.Markdown(label="Prediction Result")

    btn = gr.Button("💬 Predict My Risk")
    btn.click(fn=predict_risk, inputs=[age, bmi, avg_glucose, sex, smoker, region], outputs=output)

app.launch()

