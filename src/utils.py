import pandas as pd
from sklearn.model_selection import train_test_split


def ucitaj_podatke(putanja="data/BMW_Car_Sales_Classification.csv"):
    """Učitava sirove podatke iz CSV-a, bez ikakvih izmena."""
    df = pd.read_csv(putanja)
    return df

def pripremi_podatke(df):
    """
    Prima sirovi DataFrame i vraća enkodiranu verziju spremnu za model.
    - Izbacuje Sales_Volume (data leakage)
    - Binarno kodira Transmission i Sales_Classification
    - One-Hot Encoding za Model, Region, Color, Fuel_Type
    """
    df_model = df.drop(columns=["Sales_Volume"])

    df_model["Transmission"] = df_model["Transmission"].map({"Manual": 0, "Automatic": 1})
    df_model["Sales_Classification"] = df_model["Sales_Classification"].map({"Low": 0, "High": 1})

    kategorije_ohe = ["Model", "Region", "Color", "Fuel_Type"]
    df_model = pd.get_dummies(df_model, columns=kategorije_ohe, drop_first=True)

    return df_model

def napravi_train_test_split(df_model, test_size=0.2, random_state=42):
    """Deli pripremljene podatke na X_train, X_test, y_train, y_test."""
    X = df_model.drop(columns=["Sales_Classification"])
    y = df_model["Sales_Classification"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test