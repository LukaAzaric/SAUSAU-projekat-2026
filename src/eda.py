import matplotlib.pyplot as plt
import seaborn as sns
from utils import ucitaj_podatke

df = ucitaj_podatke()

#Raspodela ciljne promenljive:
plt.figure(figsize = (6, 4))
sns.countplot(data = df, x = "Sales_Classification")
plt.title("Raspodela klasa - Sales_Classification")
plt.savefig("outputs/01_raspodela_klasa.png")
plt.close()

#Data leakage dokaz(za sales volume)
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Sales_Classification", y="Sales_Volume")
plt.title("Sales_Volume po klasama (dokaz data leakage-a)")
plt.savefig("outputs/02_sales_volume_leakage.png")
plt.close()

#raspodela numerickih atribta (histogrami)
numericke_kolone = ["Year", "Engine_Size_L", "Mileage_KM", "Price_USD"]
fig, axes = plt.subplots(2, 2, figsize = (12, 8))
for ax, kol in zip(axes.flatten(), numericke_kolone):
    sns.histplot(data=df, x=kol, kde=True, ax=ax)
    ax.set_title(f"Raspodela: {kol}")
plt.tight_layout()
plt.savefig("outputs/03_raspodele_numerickih.png")
plt.close()

#numericki atributi naspram targeta (boxplotovi)
fig, axes = plt.subplots(2, 2, figsize = (12, 8))
for ax, kol in zip(axes.flatten(), numericke_kolone):
    sns.boxplot(data=df, x="Sales_Classification", y=kol, ax=ax)
    ax.set_title(f"{kol} po klasama")
plt.tight_layout()
plt.savefig("outputs/04_numericki_vs_target.png")
plt.close()

#korelacija nunmerickih atributa
plt.figure(figsize = (8, 6))
korelacije = df[numericke_kolone + ["Sales_Volume"]].corr()
sns.heatmap(korelacije, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelacija numerickih atributa")
plt.savefig("outputs/05_korelacija.png")
plt.close()

#kategorijski atributi naspram targeta:
kategorijske_kolone = ["Model", "Region", "Color", "Fuel_Type", "Transmission"]
for kol in kategorijske_kolone:
    plt.figure(figsize = (8, 5))
    sns.countplot(data=df, x=kol, hue="Sales_Classification")
    plt.title(f"{kol} naspram Sales_Classification")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/06_{kol}_vs_target.png")
    plt.close()

print("Svi grafici su u outputs folderu!!")