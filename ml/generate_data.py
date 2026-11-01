import os
import numpy as np
import pandas as pd


# ============================================================
# LIFELENS REALISTIC ML DATA GENERATOR
# ============================================================

NUM_DAYS = 5000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------
# OUTPUT LOCATION
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "ml_training_data.csv")


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def noise(scale=1.0):
    return np.random.normal(0, scale)


# ------------------------------------------------------------
# DATE RANGE
# ------------------------------------------------------------

dates = pd.date_range(
    start="2012-01-01",
    periods=NUM_DAYS,
    freq="D"
)


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

records = []


# ------------------------------------------------------------
# INITIAL WELLNESS STATE
# ------------------------------------------------------------

previous_mood = 6.0
previous_energy = 6.0
previous_stress = 4.0
previous_productivity = 6.0

sleep_history = []
exercise_history = []
screen_history = []
study_history = []
mood_history = []
energy_history = []
stress_history = []
productivity_history = []


# ============================================================
# GENERATE CONTINUOUS WELLNESS HISTORY
# ============================================================

for i, date in enumerate(dates):

    weekday = date.weekday()

    is_weekend = weekday >= 5


    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    if is_weekend:
        sleep = np.random.normal(7.7, 0.7)
    else:
        sleep = np.random.normal(6.9, 0.65)

    # Previous stress can disturb sleep
    sleep -= max(previous_stress - 6, 0) * 0.10

    sleep = clamp(sleep, 4.5, 10.0)


    # --------------------------------------------------------
    # EXERCISE
    # --------------------------------------------------------

    if is_weekend:
        exercise = np.random.normal(40, 20)
    else:
        exercise = np.random.normal(30, 18)

    exercise = clamp(exercise, 0, 120)


    # --------------------------------------------------------
    # SCREEN TIME
    # --------------------------------------------------------

    screen = np.random.normal(5.5, 1.4)

    # Higher stress → slightly more screen usage
    screen += max(previous_stress - 5, 0) * 0.25

    # Weekends usually have more screen time
    if is_weekend:
        screen += 0.7

    screen = clamp(screen, 1.5, 11.0)


    # --------------------------------------------------------
    # STUDY HOURS
    # --------------------------------------------------------

    if is_weekend:
        study = np.random.normal(2.0, 1.0)
    else:
        study = np.random.normal(4.5, 1.3)

    study = clamp(study, 0, 9)


    # --------------------------------------------------------
    # TREND FEATURES
    # --------------------------------------------------------

    # Recent averages make the data longitudinal.
    recent_sleep = np.mean(sleep_history[-3:]) if sleep_history else sleep
    recent_exercise = (
        np.mean(exercise_history[-3:])
        if exercise_history
        else exercise
    )
    recent_screen = (
        np.mean(screen_history[-3:])
        if screen_history
        else screen
    )
    recent_study = (
        np.mean(study_history[-3:])
        if study_history
        else study
    )


    # --------------------------------------------------------
    # MOOD
    # --------------------------------------------------------

    mood = (
        2.0

        + sleep * 0.38
        + exercise * 0.018

        - screen * 0.13

        + previous_mood * 0.28

        + recent_sleep * 0.10
        + recent_exercise * 0.006
        - recent_screen * 0.04

        + noise(0.65)
    )

    # Weekend mood boost
    if is_weekend:
        mood += 0.35

    mood = clamp(mood, 1, 10)


    # --------------------------------------------------------
    # ENERGY
    # --------------------------------------------------------

    energy = (
        1.3

        + sleep * 0.55
        + exercise * 0.012

        - screen * 0.10

        + previous_energy * 0.30

        + recent_sleep * 0.12

        + noise(0.55)
    )

    energy = clamp(energy, 1, 10)


    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    stress = (
        2.0

        + screen * 0.28
        + study * 0.20

        - exercise * 0.012
        - sleep * 0.22

        + previous_stress * 0.35

        + recent_screen * 0.08
        + recent_study * 0.05

        + noise(0.55)
    )

    # Heavy study days can increase stress
    if study > 6:
        stress += 0.6

    stress = clamp(stress, 1, 10)


    # --------------------------------------------------------
    # PRODUCTIVITY
    # --------------------------------------------------------

    productivity = (
        1.2

        + study * 0.52
        + energy * 0.38
        + mood * 0.18

        - stress * 0.24
        - screen * 0.10

        + previous_productivity * 0.18

        + noise(0.65)
    )

    productivity = clamp(productivity, 1, 10)


    # --------------------------------------------------------
    # SAVE CURRENT DAY
    # --------------------------------------------------------

    records.append({
        "Date": date,

        "SleepHours": round(sleep, 2),
        "ScreenTime": round(screen, 2),
        "ExerciseMinutes": round(exercise, 2),
        "StudyHours": round(study, 2),

        "Mood": round(mood, 2),
        "Energy": round(energy, 2),
        "Stress": round(stress, 2),
        "Productivity": round(productivity, 2)
    })


    # --------------------------------------------------------
    # UPDATE HISTORY
    # --------------------------------------------------------

    sleep_history.append(sleep)
    exercise_history.append(exercise)
    screen_history.append(screen)
    study_history.append(study)

    mood_history.append(mood)
    energy_history.append(energy)
    stress_history.append(stress)
    productivity_history.append(productivity)


    # --------------------------------------------------------
    # UPDATE PREVIOUS DAY
    # --------------------------------------------------------

    previous_mood = mood
    previous_energy = energy
    previous_stress = stress
    previous_productivity = productivity


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# CREATE NEXT-DAY TARGETS
# ============================================================

df["NextDayMood"] = df["Mood"].shift(-1)
df["NextDayEnergy"] = df["Energy"].shift(-1)
df["NextDayStress"] = df["Stress"].shift(-1)
df["NextDayProductivity"] = df["Productivity"].shift(-1)


# Remove last row because tomorrow's values don't exist
df = df.dropna().reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("LIFELENS REALISTIC ML DATA GENERATOR")
print("=" * 60)

print()
print(f"Generated records : {len(df)}")
print(f"Output file       : {OUTPUT_FILE}")

print()
print("Columns:")
print("-" * 60)

for column in df.columns:
    print(f"  {column}")

print()
print("Dataset statistics:")
print("-" * 60)

print(
    df[
        [
            "SleepHours",
            "ScreenTime",
            "ExerciseMinutes",
            "StudyHours",
            "Mood",
            "Energy",
            "Stress",
            "Productivity"
        ]
    ].describe().round(2)
)

print()
print("Correlation with next-day targets:")
print("-" * 60)

features = [
    "SleepHours",
    "ScreenTime",
    "ExerciseMinutes",
    "StudyHours",
    "Mood",
    "Energy",
    "Stress",
    "Productivity"
]

targets = [
    "NextDayMood",
    "NextDayEnergy",
    "NextDayStress",
    "NextDayProductivity"
]

for target in targets:

    print(f"\n{target}")

    correlations = (
        df[features + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
    )

    for feature, value in correlations.head(5).items():
        print(f"  {feature:<20} {value:.3f}")


print()
print("=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)
print()