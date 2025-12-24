import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dados de exemplo
X = pd.DataFrame({
    "media_5": [1,2,3,4],
    "media_10": [1,2,3,4],
    "ultimo": [1,2,3,4],
    "baixo_10": [1,0,1,0]
})
y = [1,0,1,0]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "aviator_model.pkl")
