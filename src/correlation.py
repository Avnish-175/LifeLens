import pandas as pd

def get_correlation_matrix(data):
    """Calculate correlations from recent check-ins, safely handling missing data."""

    if data.empty:
        return pd.DataFrame()

    window_size = 30

    columns = [
        "SleepHours",
        "Mood",
        "Stress",
        "Energy",
        "ExerciseMinutes",
        "ScreenTime",
        "StudyHours",
        "Productivity",
    ]

    # Keep only columns that exist in the dataset.
    available_columns = [
        column for column in columns
        if column in data.columns
    ]

    if len(available_columns) < 2:
        return pd.DataFrame()

    # Analyse the most recent 30 entries.
    recent_data = data.tail(window_size).copy()

    # Convert values to numeric; invalid or missing answers become NaN.
    numeric_data = recent_data[available_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Require at least 3 paired observations for a correlation.
    correlation = numeric_data.corr(min_periods=3)

    # Remove correlations that cannot be calculated.
    # The diagonal is retained for the heatmap.
    return correlation
