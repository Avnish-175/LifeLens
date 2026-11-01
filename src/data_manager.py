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


def save_data(new_entry):
    """
    Save a new check-in to the CSV file.
    """

    if os.path.exists(CSV_FILE):
        existing_data = pd.read_csv(CSV_FILE)
        updated_data = pd.concat(
            [existing_data, new_entry],
            ignore_index=True
        )
    else:
        updated_data = new_entry

    updated_data.to_csv(CSV_FILE, index=False)