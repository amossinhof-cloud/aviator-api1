import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Se o arquivo aviator_model.pkl não existir, cria ele
if not os.path.exists("aviator_model.pkl"):
    import train  # Executa train.py e cria o modelo

app = Flask(__name__)
model = joblib.load("aviator_model.pkl")

@app.route("/")
def home():
    return {"status": "API Aviator Online"}

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
