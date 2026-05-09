# migrate_logs.py — pokreni jednom
import os
import json

LOG_DIR = "logs"

for file in os.listdir(LOG_DIR):
    if not file.endswith(".json"):
        continue

    path = os.path.join(LOG_DIR, file)

    with open(path, "r") as f:
        try:
            data = json.load(f)
        except:
            continue

    if len(data) == 0:
        continue

    # Preskoči ako već ima nova polja
    if "asymmetry" in data[0]:
        continue

    changed = False
    for entry in data:
        # Dodaj nedostajuća polja sa default vrijednostima
        if "asymmetry" not in entry:
            entry["asymmetry"] = 0.0
            changed = True
        if "heading_error" not in entry:
            entry["heading_error"] = 0.0
            changed = True

    if changed:
        with open(path, "w") as f:
            json.dump(data, f)

print("Migracija završena.")