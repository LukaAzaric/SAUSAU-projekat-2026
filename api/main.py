from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json

#ucitavanje modela i kolona
model = joblib.load("models/model.pkl")

with open("models/kolone_modela.json", "r") as f:
    kolone_modela = json.load(f)

app = FastAPI(title="BMW Sales Classification API")

#sema ulaznih podataka
class Automobil(BaseModel):
    Model: str
    Year: int
    Region: str
    Color: str
    Fuel_Type: str
    Transmission: str
    Engine_Size_L: float
    Mileage_KM: int
    Price_USD: int

# f-ja za enkodiranje (isti postupak kao u treningu)
def enkoduj_unos(automobil: Automobil):
    df = pd.DataFrame([automobil.model_dump()])

    df["Transmission"] = df["Transmission"].map({"Manual": 0, "Automatic": 1})

    kategorije_ohe = ["Model", "Region", "Color", "Fuel_Type"]
    df = pd.get_dummies(df, columns=kategorije_ohe)
    #Poravnavanje sa kolonama koje je model video na treningu:
    #dodaje kolone koje nedostaju (popunjava sa 0)
    #uklanja eventualne visak kolone
    #postavlja isti redosled kolona kao pri treningu
    df = df.reindex(columns=kolone_modela, fill_value=0)

    return df



#endpoint za predikciju
@app.post("/predict")
def predict(automobil: Automobil):
    X_novo = enkoduj_unos(automobil)

    predikcija = model.predict(X_novo)[0]
    verovatnoca = model.predict_proba(X_novo)[0]

    klasa = "High" if predikcija == 1 else "Low"

    return {
        "Sales_Classification": klasa,
        "verovatnoca_Low": round(float(verovatnoca[0]), 4),
        "verovatnoca_High": round(float(verovatnoca[1]), 4),
    }

@app.get("/")
def root():
    return {"poruka": "BMW Sales Classification API je aktivan. Posalji POST na /predict"}