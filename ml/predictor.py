
import os
import joblib
import pandas as pd
from src.wellness_score import calculate_wellness_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


MODEL_FILES = {
    "Mood": "mood_model.pkl",
    "Energy": "energy_model.pkl",
    "Stress": "stress_model.pkl",
    "Productivity": "productivity_model.pkl",
    "SleepHours": "sleep_model.pkl",
    "ScreenTime": "screen_time_model.pkl",
    "ExerciseMinutes": "exercise_model.pkl",
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
        Build the exact 34-feature structure used by the trained ML models.
        """

        df = data.copy()

        # ---------------------------------------------------------
        # DATE / CALENDAR FEATURES
        # ---------------------------------------------------------

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.sort_values("Date").reset_index(drop=True)

        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

        # ---------------------------------------------------------
        # EXISTING ROLLING FEATURES
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # LAG FEATURES
        # ---------------------------------------------------------

        df["Lag1_SleepHours"] = df["SleepHours"].shift(1)
        df["Lag2_SleepHours"] = df["SleepHours"].shift(2)

        df["Lag1_ScreenTime"] = df["ScreenTime"].shift(1)
        df["Lag2_ScreenTime"] = df["ScreenTime"].shift(2)

        df["Lag1_ExerciseMinutes"] = df["ExerciseMinutes"].shift(1)
        df["Lag2_ExerciseMinutes"] = df["ExerciseMinutes"].shift(2)

        df["Lag1_StudyHours"] = df["StudyHours"].shift(1)
        df["Lag2_StudyHours"] = df["StudyHours"].shift(2)

        df["Lag1_Mood"] = df["Mood"].shift(1)
        df["Lag2_Mood"] = df["Mood"].shift(2)

        df["Lag1_Energy"] = df["Energy"].shift(1)
        df["Lag2_Energy"] = df["Energy"].shift(2)

        df["Lag1_Stress"] = df["Stress"].shift(1)
        df["Lag2_Stress"] = df["Stress"].shift(2)

        df["Lag1_Productivity"] = df["Productivity"].shift(1)
        df["Lag2_Productivity"] = df["Productivity"].shift(2)

        # ---------------------------------------------------------
        # ADDITIONAL ROLLING FEATURES
        # ---------------------------------------------------------

        df["Rolling3d_Mood"] = (
            df["Mood"]
            .rolling(3, min_periods=1)
            .mean()
        )

        df["Rolling3d_Energy"] = (
            df["Energy"]
            .rolling(3, min_periods=1)
            .mean()
        )

        df["Rolling3d_Productivity"] = (
            df["Productivity"]
            .rolling(3, min_periods=1)
            .mean()
        )

        df["Rolling3d_ScreenTime"] = (
            df["ScreenTime"]
            .rolling(3, min_periods=1)
            .mean()
        )

        # ---------------------------------------------------------
        # HANDLE LAG VALUES
        # ---------------------------------------------------------

        # The first two rows can have missing lag values.
        # Fill them using the current day's value so prediction
        # never receives NaN values.
        lag_columns = [
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
        ]

        for column in lag_columns:
            df[column] = df[column].fillna(df[column].iloc[-1])

        # ---------------------------------------------------------
        # FINAL NaN SAFETY
        # ---------------------------------------------------------
        # Make sure none of the 34 ML features contain NaN
        # before they are passed to the trained models.

        feature_columns = [
            "SleepHours",
            "ScreenTime",
            "ExerciseMinutes",
            "StudyHours",
            "Mood",
            "Energy",
            "Stress",
            "Productivity",
            "DayOfWeek",
            "IsWeekend",
            "Rolling3d_SleepHours",
            "Rolling7d_SleepHours",
            "Rolling3d_ExerciseMinutes",
            "Rolling3d_Stress",
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
            "Rolling3d_Mood",
            "Rolling3d_Energy",
            "Rolling3d_Productivity",
            "Rolling3d_ScreenTime",
        ]

        for column in feature_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].ffill().bfill()

        return df


def predict_next_day_ml(data):
    """
    Generate ML predictions for tomorrow.

    Returns:
        dictionary containing predicted:
        SleepHours,
        Mood,
        Stress,
        Energy,
        ExerciseMinutes,
        ScreenTime,
        Productivity.
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

            # -------------------------------------------------
            # USE EXACT FEATURE ORDER FROM TRAINED MODEL
            # -------------------------------------------------

            if hasattr(model, "feature_names_in_"):

                required_features = list(
                    model.feature_names_in_
                )

                # Make sure every required feature exists.
                missing_features = [
                    column
                    for column in required_features
                    if column not in latest.columns
                ]

                if missing_features:
                    raise ValueError(
                        f"Missing ML features for {target}: "
                        f"{missing_features}"
                    )

                X = latest[required_features]

            else:

                # Fallback feature order
                feature_columns = [
                    "SleepHours",
                    "ScreenTime",
                    "ExerciseMinutes",
                    "StudyHours",
                    "Mood",
                    "Energy",
                    "Stress",
                    "Productivity",
                    "DayOfWeek",
                    "IsWeekend",
                    "Rolling3d_SleepHours",
                    "Rolling7d_SleepHours",
                    "Rolling3d_ExerciseMinutes",
                    "Rolling3d_Stress",
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
                    "Rolling3d_Mood",
                    "Rolling3d_Energy",
                    "Rolling3d_Productivity",
                    "Rolling3d_ScreenTime",
                ]

                X = latest[feature_columns]

            # -------------------------------------------------
            # MODEL PREDICTION
            # -------------------------------------------------

            prediction = float(model.predict(X)[0])

            # -------------------------------------------------
            # TARGET-SPECIFIC LIMITS
            # -------------------------------------------------

            if target in [
                "Mood",
                "Energy",
                "Stress",
                "Productivity",
            ]:
                prediction = max(1, min(10, prediction))

            elif target == "SleepHours":
                prediction = max(4.5, min(10, prediction))

            elif target == "ScreenTime":
                prediction = max(1, min(12, prediction))

            elif target == "ExerciseMinutes":
                prediction = max(0, min(150, prediction))

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
