import os
import pandas as pd
from datetime import datetime, timedelta

PREDICTIONS_FILE = "data/historical_predictions.csv"
METRICS = ["Mood", "Energy", "Stress", "Productivity", "SleepHours", "ScreenTime", "ExerciseMinutes"]


def log_prediction(prediction_dict, prediction_date, target_date):
    """
    Safely append or update a prediction for a specific TargetDate.
    Enforces one prediction record per TargetDate.
    """
    if not prediction_dict or not prediction_date or not target_date:
        return

    pred_date_str = pd.to_datetime(prediction_date).strftime("%Y-%m-%d")
    target_date_str = pd.to_datetime(target_date).strftime("%Y-%m-%d")

    row = {
        "PredictionDate": pred_date_str,
        "TargetDate": target_date_str,
    }
    for m in METRICS:
        row[m] = prediction_dict.get(m)

    new_df = pd.DataFrame([row])

    if os.path.exists(PREDICTIONS_FILE):
        try:
            existing = pd.read_csv(PREDICTIONS_FILE)
            existing["TargetDate"] = pd.to_datetime(existing["TargetDate"]).dt.strftime("%Y-%m-%d")
            # Remove any existing prediction for this target date
            existing = existing[existing["TargetDate"] != target_date_str]
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined = combined.sort_values("TargetDate")
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    combined.to_csv(PREDICTIONS_FILE, index=False)


def load_predictions():
    """
    Load all logged predictions.
    """
    if os.path.exists(PREDICTIONS_FILE):
        try:
            df = pd.read_csv(PREDICTIONS_FILE)
            df["PredictionDate"] = pd.to_datetime(df["PredictionDate"]).dt.strftime("%Y-%m-%d")
            df["TargetDate"] = pd.to_datetime(df["TargetDate"]).dt.strftime("%Y-%m-%d")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def ensure_historical_backfill(actual_df):
    """
    If historical predictions are empty, backfills predictions for historical days
    using the available ML/base predictor from day i-1 predicting day i.
    """
    preds = load_predictions()
    if not preds.empty:
        return preds

    if actual_df is None or len(actual_df) < 3:
        return pd.DataFrame()

    from ml.predictor import predict_next_day_ml, ml_available
    from src.forecast import predict_next_day

    sorted_df = actual_df.copy()
    sorted_df["Date"] = pd.to_datetime(sorted_df["Date"]).dt.strftime("%Y-%m-%d")
    sorted_df = sorted_df.sort_values("Date").drop_duplicates(subset=["Date"])

    # Backfill for days with at least 3 prior records
    for i in range(3, len(sorted_df)):
        history_slice = sorted_df.iloc[:i]
        pred_date = history_slice.iloc[-1]["Date"]
        target_date = sorted_df.iloc[i]["Date"]

        try:
            pred = predict_next_day_ml(history_slice) if ml_available() else predict_next_day(history_slice)
            if pred:
                log_prediction(pred, pred_date, target_date)
        except Exception:
            pass

    return load_predictions()


def get_actual_vs_predicted_data(actual_df):
    """
    Matches predictions with actual check-ins strictly where actual_df['Date'] == preds['TargetDate'].
    Never compares different target dates. Never presents missing as zero.
    Returns:
      - matched_df: DataFrame with TargetDate, PredictionDate, Metric, Predicted, Actual, Error, Difference
      - summary_metrics: dict of MAE and comparison stats per metric
    """
    if actual_df is None or actual_df.empty:
        return pd.DataFrame(), {}

    preds = load_predictions()
    if preds.empty:
        preds = ensure_historical_backfill(actual_df)

    if preds.empty:
        return pd.DataFrame(), {}

    act = actual_df.copy()
    act["Date"] = pd.to_datetime(act["Date"]).dt.strftime("%Y-%m-%d")

    # Inner join on Date == TargetDate
    merged = pd.merge(preds, act, left_on="TargetDate", right_on="Date", suffixes=("_pred", "_actual"))
    if merged.empty:
        return pd.DataFrame(), {}

    rows = []
    for _, row in merged.iterrows():
        t_date = row["TargetDate"]
        p_date = row["PredictionDate"]
        for m in METRICS:
            pred_val = row.get(f"{m}_pred")
            act_val = row.get(f"{m}_actual")
            if pd.notna(pred_val) and pd.notna(act_val):
                pred_v = float(pred_val)
                act_v = float(act_val)
                diff = act_v - pred_v
                err = abs(diff)
                rows.append({
                    "TargetDate": t_date,
                    "PredictionDate": p_date,
                    "Metric": m,
                    "Predicted": pred_v,
                    "Actual": act_v,
                    "Difference": round(diff, 2),
                    "Error": round(err, 2),
                })

    result_df = pd.DataFrame(rows)
    summary = {}
    if not result_df.empty:
        for m in METRICS:
            m_df = result_df[result_df["Metric"] == m]
            if not m_df.empty:
                summary[m] = {
                    "MAE": round(m_df["Error"].mean(), 2),
                    "Count": len(m_df),
                    "MeanActual": round(m_df["Actual"].mean(), 2),
                    "MeanPredicted": round(m_df["Predicted"].mean(), 2),
                }

    return result_df, summary

