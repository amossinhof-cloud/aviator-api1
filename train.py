import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dados de exemplo para gerar o modelo
data = {
    "media_5": [1.5, 2.1, 1.8, 2.5, 1.2],
    "media_10": [1.6, 2.0, 1.9, 2.3, 1.3],
    "ultimo": [1.2, 2.5, 1.1, 3.0, 1.0],
    "baixo_10": [4, 2, 3, 1, 5],
    "alvo": [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["media_5", "media_10", "ultimo", "baixo_10"]]
y = df["alvo"]

model = RandomForestClassifier()
model.fit(X, y)

# Cria o arquivo aviator_model.pkl
joblib.dump(model, "aviator_model.pkl")
print("Modelo criado com sucesso!")
