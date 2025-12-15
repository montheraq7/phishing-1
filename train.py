import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv("Phishing_Legitimate_full.csv")

# ✅ auto-pick label column
LABEL_COL = "CLASS_LABEL" if "CLASS_LABEL" in df.columns else "label"

feature_columns = [c for c in df.columns if c not in ["id", LABEL_COL]]

df["text_combined"] = df[feature_columns].astype(str).agg(" ".join, axis=1)
df["clean_text"] = df["text_combined"].apply(clean_text)

X = df["clean_text"]
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

model = RandomForestClassifier(
    n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X_train_vec, y_train)

joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("Saved: model.joblib + vectorizer.joblib")
