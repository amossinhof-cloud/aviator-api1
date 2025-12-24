from flask import Flask, request, jsonify
from flask_cors import CORS
import os, joblib, pandas as pd

app = Flask(__name__)
CORS(app)

# Carregar ou criar modelo
if not os.path.exists("aviator_model.pkl"):
    import train  # train.py deve criar aviator_model.pkl
model = joblib.load("aviator_model.pkl")

@app.route("/")
def home():
    return {"status":"API Aviator Online"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    entrada = pd.DataFrame([{
        "media_5": data["media_5"],
        "media_10": data["media_10"],
        "ultimo": data["ultimo"],
        "baixo_10": data["baixo_10"]
    }])
    prob = model.predict_proba(entrada)[0][1]
    sinal = "ENTRAR" if prob >= 0.65 else "NÃO ENTRAR"
    return jsonify({
        "sinal": sinal,
        "probabilidade": round(float(prob), 2),
        "cashout": 2.0
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
