import os
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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
print("=" * 70)
print("LIFELENS ML MODEL TRAINING - IMPROVED 7 MODEL PIPELINE")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# REQUIRED FEATURES
# ============================================================

FEATURES = [

    # Current-day features
    "SleepHours",
    "ScreenTime",
    "ExerciseMinutes",
    "StudyHours",
    "Mood",
    "Energy",
    "Stress",
    "Productivity",

    # Calendar features
    "DayOfWeek",
    "IsWeekend",

    # Existing rolling features
    "Rolling3d_SleepHours",
    "Rolling7d_SleepHours",
    "Rolling3d_ExerciseMinutes",
    "Rolling3d_Stress",

    # Lag features
    "Lag1_SleepHours",
    "Lag2_SleepHours",

    "Lag1_ScreenTime",
    "Lag2_ScreenTime",

    "Lag1_ExerciseMinutes",
    "Lag2_ExerciseMinutes",

    "Lag1_StudyHours",
    "Lag2_StudyHours",

    "Lag1_Mood",
    "Lag2_Mood",

    "Lag1_Energy",
    "Lag2_Energy",

    "Lag1_Stress",
    "Lag2_Stress",

    "Lag1_Productivity",
    "Lag2_Productivity",

    # Additional rolling features
    "Rolling3d_Mood",
    "Rolling3d_Energy",
    "Rolling3d_Productivity",
    "Rolling3d_ScreenTime"
]


# ============================================================
# TARGETS
# ============================================================

TARGETS = {

    "mood": "NextDayMood",

    "energy": "NextDayEnergy",

    "stress": "NextDayStress",

    "productivity": "NextDayProductivity",

    "sleep": "NextDaySleepHours",

    "screen_time": "NextDayScreenTime",

    "exercise": "NextDayExerciseMinutes"
}


# ============================================================
# VALIDATE DATASET
# ============================================================

print("\nValidating dataset...")

required_columns = FEATURES + list(TARGETS.values())

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing required columns:")

    for column in missing_columns:
        print(f"  - {column}")

    raise ValueError(
        "Dataset does not contain all required ML columns."
    )


if df.empty:

    raise ValueError(
        "Dataset is empty."
    )


if df[required_columns].isnull().any().any():

    print("\nERROR: Missing values detected.")

    missing_counts = (
        df[required_columns]
        .isnull()
        .sum()
    )

    print(
        missing_counts[
            missing_counts > 0
        ]
    )

    raise ValueError(
        "Dataset contains missing values."
    )


print("Dataset validation: PASSED")

print(f"Rows available: {len(df)}")
print(f"Features used : {len(FEATURES)}")
print(f"Targets       : {len(TARGETS)}")


# ============================================================
# SORT BY DATE
# ============================================================

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[FEATURES]

print()
print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

print(
    f"\nUsing {len(FEATURES)} ML features."
)

print("\nFeature groups:")

print("  Current-day features : 8")
print("  Calendar features    : 2")
print("  Existing rolling     : 4")
print("  Lag features         : 16")
print("  Additional rolling   : 4")


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 70)

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]


print(
    f"\nTraining records: {len(X_train)}"
)

print(
    f"Testing records : {len(X_test)}"
)


if "Date" in df.columns:

    print(
        f"Training period: "
        f"{df['Date'].iloc[0].date()} "
        f"to "
        f"{df['Date'].iloc[split_index - 1].date()}"
    )

    print(
        f"Testing period : "
        f"{df['Date'].iloc[split_index].date()} "
        f"to "
        f"{df['Date'].iloc[-1].date()}"
    )


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

MODEL_CONFIGS = {

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        loss="huber",
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=42
    )
}


# ============================================================
# TARGET-SPECIFIC CLIPPING
# ============================================================

def clip_predictions(
    predictions,
    model_name
):

    if model_name in {
        "mood",
        "energy",
        "stress",
        "productivity"
    }:

        return predictions.clip(
            1,
            10
        )

    elif model_name == "sleep":

        return predictions.clip(
            4.5,
            10
        )

    elif model_name == "screen_time":

        return predictions.clip(
            1,
            12
        )

    elif model_name == "exercise":

        return predictions.clip(
            0,
            150
        )

    return predictions


# ============================================================
# TRAIN ALL TARGETS
# ============================================================

final_models = {}

final_results = {}


for target_name, target_column in TARGETS.items():

    print()
    print("=" * 70)
    print(
        f"TARGET: {target_name.upper()} "
        f"({target_column})"
    )
    print("=" * 70)

    y = df[target_column]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]


    model_results = {}


    # --------------------------------------------------------
    # TRY EVERY ALGORITHM
    # --------------------------------------------------------

    for algorithm_name, model_template in MODEL_CONFIGS.items():

        print()
        print(
            f"Training {algorithm_name}..."
        )

        model = model_template

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        predictions = clip_predictions(
            predictions,
            target_name
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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


        model_results[algorithm_name] = {

            "model": model,

            "MAE": mae,

            "RMSE": rmse,

            "R2": r2
        }


        print(
            f"  MAE : {mae:.3f}"
        )

        print(
            f"  RMSE: {rmse:.3f}"
        )

        print(
            f"  R²  : {r2:.3f}"
        )


    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_algorithm = max(
        model_results,
        key=lambda name:
            model_results[name]["R2"]
    )

    best_result = model_results[
        best_algorithm
    ]

    best_model = best_result[
        "model"
    ]


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print()
    print(
        "BEST MODEL:"
    )

    print(
        f"  {best_algorithm}"
    )

    print()
    print(
        "BEST MODEL PERFORMANCE"
    )

    print(
        f"  MAE : "
        f"{best_result['MAE']:.3f}"
    )

    print(
        f"  RMSE: "
        f"{best_result['RMSE']:.3f}"
    )

    print(
        f"  R²  : "
        f"{best_result['R2']:.3f}"
    )


    if hasattr(
        best_model,
        "feature_importances_"
    ):

        importance = pd.Series(

            best_model.feature_importances_,

            index=FEATURES

        ).sort_values(
            ascending=False
        )

        print()
        print(
            "TOP PREDICTIVE FEATURES"
        )

        print(
            "-" * 45
        )

        for feature, value in (
            importance.head(10).items()
        ):

            print(
                f"{feature:<30}"
                f"{value:.3f}"
            )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    model_file = os.path.join(
        MODEL_DIR,
        f"{target_name}_model.pkl"
    )

    joblib.dump(
        best_model,
        model_file
    )

    print()
    print(
        f"Saved best model: "
        f"{model_file}"
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    final_models[
        target_name
    ] = best_algorithm

    final_results[
        target_name
    ] = {

        "Algorithm":
            best_algorithm,

        "MAE":
            best_result["MAE"],

        "RMSE":
            best_result["RMSE"],

        "R2":
            best_result["R2"]
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("7-MODEL LIFELENS TRAINING COMPLETE")
print("=" * 70)

print()
print(
    "BEST MODEL PERFORMANCE SUMMARY"
)

print("-" * 70)

print(
    f"{'Target':<15}"
    f"{'Algorithm':<22}"
    f"{'MAE':<10}"
    f"{'RMSE':<10}"
    f"{'R²':<10}"
)

print("-" * 70)


for target_name, result in final_results.items():

    print(
        f"{target_name:<15}"
        f"{result['Algorithm']:<22}"
        f"{result['MAE']:<10.3f}"
        f"{result['RMSE']:<10.3f}"
        f"{result['R2']:<10.3f}"
    )


print()
print(
    "MODELS CREATED"
)

print("-" * 70)

for target_name in TARGETS:

    print(
        f"  ml/models/"
        f"{target_name}_model.pkl"
    )


print()
print("=" * 70)
print("TRAINING FINISHED SUCCESSFULLY")
print("=" * 70)