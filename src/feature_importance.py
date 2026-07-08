import matplotlib.pyplot as plt
import pandas as pd
from utils import ucitaj_podatke, pripremi_podatke, napravi_train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

df = ucitaj_podatke()
df_model = pripremi_podatke(df)
X_train, X_test, y_train, y_test = napravi_train_test_split(df_model)

# Treniranje modela (isti parametri kao najbolji iz tuning-a)
model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
model.fit(X_train, y_train)

#izvlacenje feature importance-a
vaznosti = pd.Series(model.feature_importances_, index=X_train.columns)
vaznosti = vaznosti.sort_values(ascending=False)

print("Feature importance (top 15):")
print(vaznosti.head(15))

#grafik
plt.figure(figsize=(10, 8))
vaznosti.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Najznacajniji atributi (Random Forest)")
plt.xlabel("Vaznost (importance)")
plt.tight_layout()
plt.savefig("outputs/07_feature_importance.png")
plt.close()

print("\nGrafik sacuvan u outputs/07_feature_importance.png")

#Svi atributi VS top n atributa:
model_svi = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
model_svi.fit(X_train, y_train)
y_pred_svi = model_svi.predict(X_test)

print("--- Model sa SVIM atributima ---")
print("Broj atributa:", X_train.shape[1])
print("Tacnost:", accuracy_score(y_test, y_pred_svi))
print("F1-score:", f1_score(y_test, y_pred_svi))

# Uzimamo top 8 najznacajnijih atributa (iz vaznosti koje smo vec izracunali)
top_atributi = vaznosti.head(8).index.tolist()
print("\nTop 8 atributa:", top_atributi)

X_train_top = X_train[top_atributi]
X_test_top = X_test[top_atributi]

model_top = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
model_top.fit(X_train_top, y_train)
y_pred_top = model_top.predict(X_test_top)

print("\n--- Model sa TOP 8 atributima ---")
print("Broj atributa:", X_train_top.shape[1])
print("Tacnost:", accuracy_score(y_test, y_pred_top))
print("F1-score:", f1_score(y_test, y_pred_top))