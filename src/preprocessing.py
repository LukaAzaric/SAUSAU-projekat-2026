import pandas as pd
from sklearn.model_selection import train_test_split

#ucitavamo podatke:
df = pd.read_csv("data/BMW_Car_Sales_Classification.csv")
print("Oblik dataseta: ", df.shape)
print("\nTipovi podataka: ", df.dtypes)

#da li neke vrednosti nedostaju?
print("\nNeodstajuce vrednosti po koloni: ")
print(df.isnull().sum())

#proveravamo anomalije
print("\nOsnovna statistika numerickih kolona: ")
print(df.describe())

#provera ekstremnih vrednosti
kategorije = ["Model", "Region", "Color", "Fuel_Type", "Transmission"]
for kol in kategorije:
    print(f"\n{kol} - jedinstvene vrednosti ({df[kol].nunique()}):")
    print(df[kol].unique())

#provera raspodele ciljne promenljive
print("\nRaspodela Sales_Classification: ")
print(df["Sales_Classification"].value_counts())
print(df["Sales_Classification"].value_counts(normalize = True) * 100)

#Sales_Volume vs Sales_Classification
print("\nSales_Volume po klasama:")
print(df.groupby("Sales_Classification")["Sales_Volume"].describe())

print("\nMin i max Sales_Volume po klasi:")
print(df.groupby("Sales_Classification")["Sales_Volume"].agg(["min", "max"]))

#ENKODIRANJE PODATAKA

#Izbacujemo Sales_Volume iz prediktora (data leakage - direktno određuje target)
df_model = df.drop(columns=["Sales_Volume"])

#Binarno kodiranje Transmission-a (Manual=0, Automatic=1)
df_model["Transmission"] = df_model["Transmission"].map({"Manual": 0, "Automatic": 1})

#Binarno kodiranje targeta (Low=0, High=1)
df_model["Sales_Classification"] = df_model["Sales_Classification"].map({"Low": 0, "High": 1})

#One-Hot Encoding za Model, Region, Color, Fuel_Type
kategorije_ohe = ["Model", "Region", "Color", "Fuel_Type"]
df_model = pd.get_dummies(df_model, columns=kategorije_ohe, drop_first=True)

print("\nOblik nakon enkodiranja:", df_model.shape)
print(df_model.head())

#Podela na X i y
X = df_model.drop(columns=["Sales_Classification"])
y = df_model["Sales_Classification"]
print("\nOblik X (prediktori):", X.shape)
print("Oblik y (target):", y.shape)

#train/test podela
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain set:", X_train.shape)
print("Test set:", X_test.shape)