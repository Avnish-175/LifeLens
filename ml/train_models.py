import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "ml_training_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("LIFELENS ML MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "SleepHours",
    "ScreenTime",
    "ExerciseMinutes",
    "StudyHours",
    "Mood",
    "Energy",
    "Stress",
    "Productivity"
]


# ============================================================
# TARGETS
# ============================================================

TARGETS = {
    "mood": "NextDayMood",
    "energy": "NextDayEnergy",
    "stress": "NextDayStress",
    "productivity": "NextDayProductivity"
}


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[FEATURES]


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# We deliberately do NOT randomly shuffle the data.
#
# Earlier records → training
# Later records  → testing
#
# This is more appropriate for time-based prediction.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]


print(f"\nTraining records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ============================================================
# TRAIN MODELS
# ============================================================

results = {}


for model_name, target_column in TARGETS.items():

    print()
    print("-" * 60)
    print(f"Training {model_name.upper()} model")
    print("-" * 60)

    y = df[target_column]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(X_test)

    # Keep predictions inside the LifeLens 1-10 scale

    predictions = predictions.clip(1, 10)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(
        ascending=False
    )

    print("\nTop factors:")

    for feature, value in importance.head(5).items():

        print(
            f"  {feature:<20} "
            f"{value:.3f}"
        )

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    model_file = os.path.join(
        MODEL_DIR,
        f"{model_name}_model.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    print(
        f"\nSaved model: {model_file}"
    )

    results[model_name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nModel Performance:")
print()

for model_name, metrics in results.items():

    print(
        f"{model_name.capitalize():<15}"
        f"MAE={metrics['MAE']:.3f}   "
        f"RMSE={metrics['RMSE']:.3f}   "
        f"R²={metrics['R2']:.3f}"
    )


print()
print("Models created:")

for model_name in TARGETS:

    print(
        f"  ml/models/{model_name}_model.pkl"
    )

print()
print("=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)