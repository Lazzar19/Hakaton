# =========================================================
# FILTER METADRIVE JSON LOGOVA
# =========================================================
#
# OVAJ SCRIPT:
#
# 1. Briše katastrofalne epizode
# 2. Briše kratke epizode
# 3. Briše epizode sa previše crash-eva
# 4. Briše epizode sa premalo distance
# 5. Briše epizode gdje auto stoji
# 6. Briše NaN / korumpirane podatke
# 7. Zadržava samo kvalitetne vožnje
#
# =========================================================

import os
import json
import shutil
import numpy as np

LOG_DIR = "logs"

# Folder gdje će biti prebačeni loši logovi
BAD_DIR = os.path.join(LOG_DIR, "filtered_out")

os.makedirs(BAD_DIR, exist_ok=True)

# =========================================================
# PARAMETRI FILTRIRANJA
# =========================================================

MIN_FRAMES = 150

MIN_TOTAL_REWARD = -10.0

MIN_AVG_SPEED = 0.08

MAX_CRASH_RATIO = 0.15

MAX_OUT_OF_ROAD_RATIO = 0.10

MIN_MOVEMENT_FRAMES = 50

MAX_ZERO_SPEED_RATIO = 0.60

# =========================================================
# HELPERS
# =========================================================

def safe_float(v, default=0.0):
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except:
        return default


def is_corrupted(entry):

    required = [
        "speed",
        "lane_offset",
        "reward"
    ]

    for r in required:
        if r not in entry:
            return True

    vals = [
        entry.get("speed", 0),
        entry.get("lane_offset", 0),
        entry.get("reward", 0),
    ]

    for v in vals:
        if np.isnan(safe_float(v)):
            return True

    return False


# =========================================================
# MAIN FILTER
# =========================================================

kept_files = 0
removed_files = 0

for file_name in os.listdir(LOG_DIR):

    if not file_name.endswith(".json"):
        continue

    path = os.path.join(LOG_DIR, file_name)

    try:

        with open(path, "r") as f:
            data = json.load(f)

        # -------------------------------------------------
        # PRAZAN / PREKRATAK FILE
        # -------------------------------------------------

        if not isinstance(data, list):
            raise Exception("JSON nije lista")

        if len(data) < MIN_FRAMES:

            shutil.move(
                path,
                os.path.join(BAD_DIR, file_name)
            )

            removed_files += 1
            print(f"[REMOVE] {file_name} | prekratka epizoda")
            continue

        # -------------------------------------------------
        # STATISTIKE
        # -------------------------------------------------

        total_reward = 0.0
        avg_speed = 0.0

        crash_count = 0
        out_count = 0

        zero_speed_frames = 0
        movement_frames = 0

        corrupted = False

        for step in data:

            if is_corrupted(step):
                corrupted = True
                break

            speed = safe_float(step.get("speed", 0))
            reward = safe_float(step.get("reward", 0))

            total_reward += reward
            avg_speed += speed

            if speed < 0.02:
                zero_speed_frames += 1
            else:
                movement_frames += 1

            if step.get("crash", False):
                crash_count += 1

            if step.get("out_of_road", False):
                out_count += 1

        if corrupted:

            shutil.move(
                path,
                os.path.join(BAD_DIR, file_name)
            )

            removed_files += 1

            print(f"[REMOVE] {file_name} | korumpiran")
            continue

        avg_speed /= len(data)

        crash_ratio = crash_count / len(data)
        out_ratio = out_count / len(data)

        zero_speed_ratio = zero_speed_frames / len(data)

        # -------------------------------------------------
        # FILTERI
        # -------------------------------------------------

        bad = False
        reason = ""

        # Loš reward
        if total_reward < MIN_TOTAL_REWARD:
            bad = True
            reason = "loš total reward"

        # Presporo
        elif avg_speed < MIN_AVG_SPEED:
            bad = True
            reason = "premala brzina"

        # Previše crash-eva
        elif crash_ratio > MAX_CRASH_RATIO:
            bad = True
            reason = "previše crash"

        # Previše izlazaka sa puta
        elif out_ratio > MAX_OUT_OF_ROAD_RATIO:
            bad = True
            reason = "previše out_of_road"

        # Auto skoro ne mrda
        elif movement_frames < MIN_MOVEMENT_FRAMES:
            bad = True
            reason = "nema kretanja"

        # Stojeći agent
        elif zero_speed_ratio > MAX_ZERO_SPEED_RATIO:
            bad = True
            reason = "previše stajanja"

        # -------------------------------------------------
        # REZULTAT
        # -------------------------------------------------

        if bad:

            shutil.move(
                path,
                os.path.join(BAD_DIR, file_name)
            )

            removed_files += 1

            print(
                f"[REMOVE] {file_name} | "
                f"{reason}"
            )

        else:

            kept_files += 1

            print(
                f"[KEEP] {file_name} | "
                f"reward={total_reward:.2f} | "
                f"speed={avg_speed:.2f}"
            )

    except Exception as e:

        try:
            shutil.move(
                path,
                os.path.join(BAD_DIR, file_name)
            )
        except:
            pass

        removed_files += 1

        print(f"[ERROR] {file_name} | {e}")

# =========================================================
# SUMMARY
# =========================================================

print("\n=================================================")
print("FILTER COMPLETE")
print("=================================================")

print(f"Zadržano: {kept_files}")
print(f"Uklonjeno: {removed_files}")

print(f"\nLoši logovi prebačeni u:")
print(BAD_DIR)