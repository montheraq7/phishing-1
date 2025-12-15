import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Phishing Demo API")

# Allow CodePen (or any frontend) to call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    cleaned = clean_text(req.text)
    X = vectorizer.transform([cleaned])
    pred = int(model.predict(X)[0])

    # Adjust labels if your dataset uses opposite meaning
    label = "phishing" if pred == 1 else "legitimate"

    return {"prediction": pred, "label": label}
