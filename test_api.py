import requests

podaci = {
    "Model": "X5",
    "Year": 2021,
    "Region": "Europe",
    "Color": "Black",
    "Fuel_Type": "Petrol",
    "Transmission": "Automatic",
    "Engine_Size_L": 3.0,
    "Mileage_KM": 45000,
    "Price_USD": 82000
}

odgovor = requests.post("http://127.0.0.1:8000/predict", json=podaci)

print("Status kod:", odgovor.status_code)
print("Odgovor:", odgovor.json())