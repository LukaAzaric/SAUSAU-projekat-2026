from utils import ucitaj_podatke, pripremi_podatke, napravi_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import json

df = ucitaj_podatke()
df_model = pripremi_podatke(df)
X_train, X_test, y_train, y_test = napravi_train_test_split(df_model)
print("Train set: ", X_train.shape)
print("Test set: ", X_test.shape)

#Baseline pogadjanje:
baseline_tacnost = y_train.value_counts(normalize=True).max()
print(f"\nBaesline tacnost(uvk predvidja samo vecinsku klasu): {baseline_tacnost: .4f}")

#Logistica regresija:
model_lr = LogisticRegression(max_iter = 1000, random_state = 42)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("\n -----Logicka regresija ----")
print("\nTacnost: ", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

#random forest:
model_rf = RandomForestClassifier(n_estimators = 200, random_state = 42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print("\n--- Random Forest ---")
print("Tacnost:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

#gradient boosting(poredimo performanse):
model_gb = GradientBoostingClassifier(random_state=42)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)

print("\n--- Gradient Boosting ---")
print("Tacnost:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

#logisticka regresija sa balansiranjem klasa:
model_lr_balanced = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
model_lr_balanced.fit(X_train, y_train)
y_pred_balanced = model_lr_balanced.predict(X_test)

print("\n--- Logisticka regresija (class_weight=balanced) ---")
print("Tacnost:", accuracy_score(y_test, y_pred_balanced))
print(classification_report(y_test, y_pred_balanced))

#podesavanje hiperparametara:
parametri_rf = {
    "n_estimators":[100, 200, 300],
    "max_depth":[5, 10, 20, None],
    "min_samples_split":[2, 5, 10]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid = parametri_rf,
                       cv = 5,
                       scoring = "f1",
                       n_jobs = -1,
                       verbose = 2)
grid_rf.fit(X_train, y_train)
print("\n----Random forest posle tuning-a----")
print("\nNajbolji parametri: ", grid_rf.best_params_)

najbolji_rf = grid_rf.best_estimator_
y_pred_najbolji_rf = najbolji_rf.predict(X_test)

print("\nTacnost: ", accuracy_score(y_test, y_pred_najbolji_rf))
print(classification_report(y_test, y_pred_najbolji_rf))


#cuvanje najboljeg modela:
joblib.dump(model_rf, "models/model.pkl")
print("\nModel sacuvan u models/model.pkl")

#FINALNI MODEL:
finalni_model = RandomForestClassifier(
    n_estimators=100, max_depth=None, min_samples_split=2, random_state=42
)
finalni_model.fit(X_train, y_train)

joblib.dump(finalni_model, "models/model.pkl")
# Cuvamo i tacan spisak kolona (redosled i nazivi) kako bismo mogli
# da poravnamo buduce ulazne podatke sa onim sto je model video na treningu
kolone_modela = X_train.columns.tolist()
with open("models/kolone_modela.json", "w") as f:
    json.dump(kolone_modela, f)

print(f"\nFinalni model sacuvan. Broj kolona: {len(kolone_modela)}")