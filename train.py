import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "training.csv"
ARTIFACT_DIR = BASE_DIR / "app" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Looking for training data at: {DATA_PATH}")


DATA_PATH = Path("data/training.csv")
ARTIFACT_DIR = Path("app/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target"
NUMERIC = ["age", "bmi", "avg_glucose"]
CATEGORICAL = ["sex", "smoker", "region"]

def main():
    df = pd.read_csv(DATA_PATH)
    keep_cols = [TARGET] + NUMERIC + CATEGORICAL
    df = df[keep_cols].dropna()

    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )

    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    print("ROC-AUC:", round(roc_auc_score(yte, proba), 3))

    joblib.dump(pipe.named_steps["pre"], ARTIFACT_DIR / "preprocessor.pkl")
    joblib.dump(pipe.named_steps["clf"], ARTIFACT_DIR / "model.pkl")
    print("Saved artifacts to app/artifacts/")

if __name__ == "__main__":
    main()
