import pandas as pd
import os

CSV_FILE = "data/daily_checkins.csv"

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


def save_data(new_entry):
    """
    Save a check-in while enforcing one check-in per date.
    If the date already exists, update that record.
    Otherwise, add a new record.
    """

    new_entry = new_entry.copy()

    # Normalize the incoming date.
    new_entry["Date"] = normalize_date_series(new_entry["Date"])

    # Reject invalid dates.
    if new_entry["Date"].isna().any():
        raise ValueError("Invalid check-in date.")

    if os.path.exists(CSV_FILE):
        existing_data = pd.read_csv(CSV_FILE)

        # Normalize all existing dates so different formats
        # are treated as the same date.
        existing_data["Date"] = normalize_date_series(existing_data["Date"])

        new_date = new_entry.iloc[0]["Date"]

        # Remove any existing record for this date.
        existing_data = existing_data[
            existing_data["Date"] != new_date
        ]

        # Add the new/updated record.
        updated_data = pd.concat(
            [existing_data, new_entry],
            ignore_index=True
        )

    else:
        updated_data = new_entry

    # Keep the CSV sorted chronologically.
    updated_data["Date"] = pd.to_datetime(
        updated_data["Date"],
        errors="coerce"
    )

    updated_data = updated_data.sort_values("Date")

    updated_data["Date"] = updated_data["Date"].dt.strftime("%Y-%m-%d")

    updated_data.to_csv(CSV_FILE, index=False)
