
import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


MODEL_FILES = {
    "Mood": "mood_model.pkl",
    "Energy": "energy_model.pkl",
    "Stress": "stress_model.pkl",
    "Productivity": "productivity_model.pkl",
}


def load_models():
    """
    Load all trained LifeLens ML models.
    """

    models = {}

    for target, filename in MODEL_FILES.items():

        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            return {}

        try:
            models[target] = joblib.load(path)
        except Exception:
            return {}

    return models


def build_features(data):
    """
    Build the exact feature structure used by the ML models.
    """

    df = data.copy()

    # Convert date
    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.sort_values("Date").reset_index(drop=True)

    # Calendar features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Rolling behavioural features
    df["Rolling3d_SleepHours"] = (
        df["SleepHours"]
        .rolling(3, min_periods=1)
        .mean()
    )

    df["Rolling7d_SleepHours"] = (
        df["SleepHours"]
        .rolling(7, min_periods=1)
        .mean()
    )

    df["Rolling3d_ExerciseMinutes"] = (
        df["ExerciseMinutes"]
        .rolling(3, min_periods=1)
        .mean()
    )

    df["Rolling3d_Stress"] = (
        df["Stress"]
        .rolling(3, min_periods=1)
        .mean()
    )

    return df


def predict_next_day_ml(data):
    """
    Generate ML predictions for tomorrow.

    Returns:
        dictionary containing predicted
        Mood, Energy, Stress and Productivity.
    """

    models = load_models()

    if not models:
        return {}

    if data.empty:
        return {}

    try:

        features = build_features(data)

        # Most recent check-in is used to predict tomorrow.
        latest = features.iloc[[-1]].copy()

        predictions = {}

        for target, model in models.items():

            # Prefer the exact feature order stored by sklearn.
            if hasattr(model, "feature_names_in_"):

                required_features = list(
                    model.feature_names_in_
                )

                # Add any missing feature safely.
                for column in required_features:
                    if column not in latest.columns:
                        latest[column] = 0

                X = latest[required_features]

            else:

                # Fallback for models without feature_names_in_
                feature_columns = [
                    "DayOfWeek",
                    "IsWeekend",
                    "SleepHours",
                    "ScreenTime",
                    "ExerciseMinutes",
                    "StudyHours",
                    "Mood",
                    "Energy",
                    "Stress",
                    "Productivity",
                    "Rolling3d_SleepHours",
                    "Rolling7d_SleepHours",
                    "Rolling3d_ExerciseMinutes",
                    "Rolling3d_Stress",
                ]

                X = latest[feature_columns]

            prediction = model.predict(X)[0]

            # Keep predictions within LifeLens' 0-10 scale.
            prediction = max(0, min(10, float(prediction)))

            predictions[target] = round(prediction, 1)

        return predictions

    except Exception as error:

        print("ML prediction error:", repr(error))

        raise


def ml_available():
    """
    Check whether all trained models are available.
    """

    return all(
        os.path.exists(
            os.path.join(MODEL_DIR, filename)
        )
        for filename in MODEL_FILES.values()
    )
