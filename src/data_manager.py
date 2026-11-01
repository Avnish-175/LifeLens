import pandas as pd
import os

CSV_FILE = "data/daily_checkins.csv"

# Numerical validation bounds
VALIDATION_RANGES = {
    "SleepHours": (0.0, 24.0),
    "ScreenTime": (0.0, 24.0),
    "ExerciseMinutes": (0.0, 1440.0),
    "StudyHours": (0.0, 24.0),
    "Mood": (1.0, 10.0),
    "Energy": (1.0, 10.0),
    "Stress": (1.0, 10.0),
    "Productivity": (1.0, 10.0),
}


def check_data_file():
    print("Current working directory:", os.getcwd())
    print("CSV path being used:", os.path.abspath(CSV_FILE))
    print("CSV file exists:", os.path.exists(CSV_FILE))

    if os.path.exists(CSV_FILE):
        check_data = pd.read_csv(CSV_FILE)
        print("Number of rows in CSV:", len(check_data))
        print("CSV columns:", check_data.columns.tolist())


def load_data():
    """
    Load all check-in data.
    """
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)

    return pd.DataFrame()


def normalize_date_series(series):
    """
    Safely parse dates in %Y-%m-%d, %d-%m-%Y, or mixed string/date formats
    and return formatted string series in %Y-%m-%d.
    """
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(s.loc[missing], format="%d-%m-%Y", errors="coerce")
    still_missing = parsed.isna()
    if still_missing.any():
        parsed.loc[still_missing] = pd.to_datetime(s.loc[still_missing], format="mixed", errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def validate_checkin_entry(entry):
    """
    Validate that check-in values fall within defined ranges.
    Raises ValueError with a user-friendly message on invalid data.
    """
    if entry is None or entry.empty:
        raise ValueError("Check-in entry cannot be empty.")

    row = entry.iloc[0]

    # Validate Date
    if "Date" not in entry.columns or pd.isna(row["Date"]):
        raise ValueError("A valid check-in date is required.")

    for field, (min_v, max_v) in VALIDATION_RANGES.items():
        if field in entry.columns:
            val = row[field]
            # Optional reflection values may be None / NaN
            if pd.notna(val) and val is not None:
                try:
                    num = float(val)
                except (ValueError, TypeError):
                    raise ValueError(f"{field} must be a valid number.")
                if num < min_v or num > max_v:
                    raise ValueError(
                        f"{field} must be between {min_v:g} and {max_v:g}. Got {num:g}."
                    )


def save_data(new_entry):
    """
    Save a check-in while enforcing one check-in per date.
    Validates input ranges, deduplicates legacy dates,
    updates if date already exists, or adds a new record.
    """
    if new_entry is None or new_entry.empty:
        raise ValueError("Cannot save empty check-in data.")

    new_entry = new_entry.copy()

    # Normalize the incoming date.
    new_entry["Date"] = normalize_date_series(new_entry["Date"])

    # Reject invalid dates.
    if new_entry["Date"].isna().any():
        raise ValueError("Invalid check-in date.")

    # Validate all fields against acceptable ranges
    validate_checkin_entry(new_entry)

    if os.path.exists(CSV_FILE):
        existing_data = pd.read_csv(CSV_FILE)

        if not existing_data.empty:
            # Normalize all existing dates so different formats are treated as the same date.
            existing_data["Date"] = normalize_date_series(existing_data["Date"])

            # Clean any pre-existing duplicate dates in the CSV, keeping the newest valid entry
            existing_data = existing_data.drop_duplicates(subset=["Date"], keep="last")

            new_date = new_entry.iloc[0]["Date"]

            # Remove any existing record for this date (to overwrite with newly submitted values).
            existing_data = existing_data[existing_data["Date"] != new_date]

            # Add the new/updated record.
            updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
        else:
            updated_data = new_entry
    else:
        updated_data = new_entry

    # Keep the CSV sorted chronologically.
    updated_data["Date"] = pd.to_datetime(updated_data["Date"], errors="coerce")
    updated_data = updated_data.sort_values("Date")
    updated_data["Date"] = updated_data["Date"].dt.strftime("%Y-%m-%d")

    # Final safeguard: deduplicate by date
    updated_data = updated_data.drop_duplicates(subset=["Date"], keep="last")

    # Ensure parent directory exists before saving
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    updated_data.to_csv(CSV_FILE, index=False)
