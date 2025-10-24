---
title: Diabetes Risk Chatbot
emoji: 🩷
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app/ui.py
pinned: false
license: mit
---

# 🩷 Diabetes Risk Chatbot

An interactive **AI assistant** built with **Gradio + Scikit-Learn** that predicts whether a diabetic patient should **visit the ER** or **schedule a doctor’s appointment**, based on simple health data.

---

## 💬 How It Works
1. Enter your information:  
   • Age  
   • BMI  
   • Average Glucose Level  
   • Sex, Smoker status, Region  
2. The model preprocesses your data using a trained pipeline.  
3. It returns a friendly message showing your predicted **risk level** 💚 ⚠️ 🚨  

---

## ⚙️ Tech Stack
| Tool | Purpose |
|------|----------|
| 🐍 Python 3.12 | Core language |
| 🤖 Scikit-Learn | Model training & prediction |
| 🎨 Gradio 5 | Interactive UI |
| 📊 Pandas / NumPy | Data handling |
| ☁️ Hugging Face Spaces | Deployment |

---

## 🧠 Model Info
Trained using healthcare data to estimate diabetic risk with a **logistic-regression pipeline**.  
Includes scaling and one-hot encoding for categorical features.

---

## 🌍 Try It Live
👉 **[Launch the Chatbot](https://huggingface.co/spaces/prettytechgirl/diabetes-risk-chatbot)**  
*(Takes a few seconds to load the first time)*  

---

## 🩵 Author
Built with care by **@prettytechgirl (Tenika Powell)**  
💬 “Sourced from code & coffee ☕ — made for learning and helping others.”  

---

### 💖 Version Notes
- v1.0 — Initial launch on Hugging Face Spaces  
- v1.1 — UI and model refinements coming soon  
