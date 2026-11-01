import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

NUM_RECORDS = 3000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "ml_training_data.csv"
)


# ============================================================
# CREATE DATA DIRECTORY IF NEEDED
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# DATE RANGE
# ============================================================

dates = pd.date_range(
    start="2018-01-01",
    periods=NUM_RECORDS,
    freq="D"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def clip(value, minimum, maximum):
    return np.clip(value, minimum, maximum)


# ============================================================
# GENERATE BASE BEHAVIOURAL DATA
# ============================================================

sleep = np.random.normal(
    loc=7.0,
    scale=1.15,
    size=NUM_RECORDS
)

sleep = clip(sleep, 4.0, 10.0)


screen_time = np.random.normal(
    loc=5.5,
    scale=2.0,
    size=NUM_RECORDS
)

screen_time = clip(screen_time, 1.0, 12.0)


exercise = np.random.normal(
    loc=35,
    scale=22,
    size=NUM_RECORDS
)

exercise = clip(exercise, 0, 120)


study = np.random.normal(
    loc=4.0,
    scale=2.0,
    size=NUM_RECORDS
)

study = clip(study, 0, 10)


# ============================================================
# GENERATE WELLNESS VARIABLES
# ============================================================

# Mood is influenced by:
# - sleep positively
# - exercise positively
# - screen time negatively
# - random variation

mood = (
    5.5
    + (sleep - 6.5) * 0.9
    + exercise * 0.015
    - screen_time * 0.12
    + np.random.normal(0, 1.1, NUM_RECORDS)
)

mood = clip(mood, 1, 10)


# Energy is influenced strongly by sleep and exercise

energy = (
    5.5
    + (sleep - 6.5) * 1.1
    + exercise * 0.012
    - screen_time * 0.08
    + np.random.normal(0, 1.0, NUM_RECORDS)
)

energy = clip(energy, 1, 10)


# Stress is influenced by:
# - excessive screen time
# - low sleep
# - study/work pressure
# - random variation

stress = (
    5.0
    + (6.5 - sleep) * 0.8
    + screen_time * 0.18
    + study * 0.08
    + np.random.normal(0, 1.0, NUM_RECORDS)
)

stress = clip(stress, 1, 10)


# Productivity is influenced by:
# - study
# - sleep
# - energy
# - mood
# - stress negatively

productivity = (
    3.5
    + study * 0.55
    + (sleep - 6.5) * 0.55
    + energy * 0.25
    + mood * 0.15
    - stress * 0.35
    - screen_time * 0.08
    + np.random.normal(0, 1.0, NUM_RECORDS)
)

productivity = clip(productivity, 1, 10)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "Date": dates,

    "SleepHours": np.round(sleep, 1),

    "ScreenTime": np.round(screen_time, 1),

    "ExerciseMinutes": np.round(exercise, 0),

    "StudyHours": np.round(study, 1),

    "Mood": np.round(mood, 1),

    "Energy": np.round(energy, 1),

    "Stress": np.round(stress, 1),

    "Productivity": np.round(productivity, 1),
})


# ============================================================
# CREATE NEXT-DAY TARGETS
# ============================================================

# These are what our ML models will eventually learn to predict.

df["NextDayMood"] = (
    df["Mood"].shift(-1)
)

df["NextDayEnergy"] = (
    df["Energy"].shift(-1)
)

df["NextDayStress"] = (
    df["Stress"].shift(-1)
)

df["NextDayProductivity"] = (
    df["Productivity"].shift(-1)
)


# ============================================================
# REMOVE LAST ROW
# ============================================================

# The final record has no "next day" available.

df = df.dropna().reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# INFORMATION
# ============================================================

print()
print("=" * 60)
print("LifeLens ML DATASET GENERATED")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("First 5 records:")
print(df.head())

print()
print("Dataset statistics:")
print(df.describe().round(2))

print()
print("=" * 60)
print("PHASE 1 COMPLETE")
print("=" * 60)