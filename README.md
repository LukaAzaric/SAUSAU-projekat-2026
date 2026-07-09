# BMW Car Sales Classification

Kako pokrenuti projekat (sve komande iz root foldera projekta):

**Aktiviraj venv**

&#x20;  `venv\\Scripts\\activate`

**Instaliraj pakete**

`pip install -r requirements.txt`

**Pokreni fajlove ovim redosledom**

&#x20;  py src/preprocessing.py

&#x20;  py src/eda.py

&#x20;  py src/train\_model.py

&#x20;  py src/feature\_importance.py

**Pokreni API**

`uvicorn api.main:app --reload`

**Otvori u browseru**: `http://127.0.0.1:8000/docs`

Tu se moze testirati `/predict` (unesi podatke o autu, klikni Execute).

Grafici se cuvaju u `outputs/`, model u `models/`. Sve objasnjeno u dokumentaciji.

