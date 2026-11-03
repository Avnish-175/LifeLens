"""
LifeLens Deterministic NLP Facts Engine
src/nlp_facts.py

Calculates authoritative mathematical facts, metrics deltas, directions,
and semantic classifications in pure Python before feeding context to the
local NLP model or Streamlit UI.

Rules:
- Python is the sole source of truth for arithmetic and classifications.
- NLP model is used strictly for explanatory prose.
- Stress is inverted: lower stress = improved, higher stress = worsening.
- Positive metrics: increase = improved, decrease = worsening.
- Zero change = unchanged.
"""

import math
from typing import Dict, Any, Optional, Union
import pandas as pd

METRIC_DISPLAY_NAMES = {
    "SleepHours": "Sleep",
    "ScreenTime": "Screen Time",
    "ExerciseMinutes": "Exercise",
    "StudyHours": "Study Hours",
    "Mood": "Mood",
    "Stress": "Stress",
    "Energy": "Energy",
    "Productivity": "Productivity",
}


def _valid_number(value: Any) -> Optional[float]:
    """Return a finite float or None for missing/invalid input."""
    try:
        if value is None:
            return None
        val = float(value)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def calculate_direction(change: float, metric: str) -> str:
    """
    Determine the qualitative direction of a numeric change.
    
    Stress is inverted:
        change < 0 -> improved
        change > 0 -> worsening
        change == 0 -> unchanged
        
    Positive metrics (Mood, Energy, Productivity, Sleep, Exercise, Wellness):
        change > 0 -> improved
        change < 0 -> worsening (or declining)
        change == 0 -> unchanged
    """
    if abs(change) < 1e-6:
        return "unchanged"

    metric_lower = metric.lower()
    if "stress" in metric_lower:
        return "improved" if change < 0 else "worsening"
    elif "screen" in metric_lower:
        # Screen time increase is generally worsening or neutral
        return "worsening" if change > 0 else "improved"
    else:
        return "improved" if change > 0 else "declining"


# =====================================================================
# STEP 3: DAILY FACTS
# =====================================================================

def build_daily_facts(
    latest: Union[pd.Series, Dict[str, Any]],
    wellness_score: Optional[float] = None,
    wellness_category: Optional[str] = None,
    history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build authoritative deterministic facts for daily wellness analysis.
    
    Returns exact values for the latest record, category, and recent delta if available.
    """
    if latest is None:
        return {}

    if isinstance(latest, pd.Series):
        latest_dict = latest.to_dict()
    else:
        latest_dict = dict(latest)

    sleep = _valid_number(latest_dict.get("SleepHours"))
    screen = _valid_number(latest_dict.get("ScreenTime"))
    exercise = _valid_number(latest_dict.get("ExerciseMinutes"))
    study = _valid_number(latest_dict.get("StudyHours"))

    mood = _valid_number(latest_dict.get("Mood"))
    stress = _valid_number(latest_dict.get("Stress"))
    energy = _valid_number(latest_dict.get("Energy"))
    productivity = _valid_number(latest_dict.get("Productivity"))

    # Compute wellness score from wellness_score module if not supplied
    if wellness_score is None:
        from src.wellness_score import calculate_wellness_score
        computed_score, computed_cat = calculate_wellness_score(
            sleep=sleep,
            mood=mood,
            stress=stress,
            energy=energy,
            exercise=exercise,
            screen_time=screen,
            productivity=productivity,
        )
        wellness_score = computed_score
        wellness_category = computed_cat
    elif wellness_category is None:
        if wellness_score >= 85:
            wellness_category = "Excellent"
        elif wellness_score >= 70:
            wellness_category = "Healthy"
        elif wellness_score >= 55:
            wellness_category = "Needs Improvement"
        else:
            wellness_category = "At Risk"

    # Previous day change if history available
    score_change = None
    prev_score_val = None
    score_direction = "unchanged"
    if history is not None and len(history) >= 2:
        try:
            prev_row = history.iloc[-2]
            prev_score = _valid_number(prev_row.get("WellnessScore"))
            if prev_score is None:
                from src.wellness_score import calculate_wellness_score
                prev_score, _ = calculate_wellness_score(
                    sleep=_valid_number(prev_row.get("SleepHours")),
                    mood=_valid_number(prev_row.get("Mood")),
                    stress=_valid_number(prev_row.get("Stress")),
                    energy=_valid_number(prev_row.get("Energy")),
                    exercise=_valid_number(prev_row.get("ExerciseMinutes")),
                    screen_time=_valid_number(prev_row.get("ScreenTime")),
                    productivity=_valid_number(prev_row.get("Productivity")),
                )
            if prev_score is not None and wellness_score is not None:
                prev_score_val = int(round(prev_score))
                curr_score_val = int(round(wellness_score))
                score_change = curr_score_val - prev_score_val
                score_direction = calculate_direction(score_change, "WellnessScore")
        except Exception:
            score_change = None

    facts = {
        "wellness_score": int(round(wellness_score)) if wellness_score is not None else None,
        "wellness_category": str(wellness_category) if wellness_category else "Fair",
        "previous_wellness_score": prev_score_val,
        "score_change": score_change,
        "score_direction": score_direction,
        "sleep_hours": round(sleep, 1) if sleep is not None else None,
        "screen_time": round(screen, 1) if screen is not None else None,
        "exercise_minutes": int(round(exercise)) if exercise is not None else None,
        "study_hours": round(study, 1) if study is not None else None,
        "mood": int(round(mood)) if mood is not None else None,
        "stress": int(round(stress)) if stress is not None else None,
        "energy": int(round(energy)) if energy is not None else None,
        "productivity": int(round(productivity)) if productivity is not None else None,
    }
    return facts


# =====================================================================
# STEP 4: WEEKLY FACTS
# =====================================================================

def build_weekly_facts(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate deterministic weekly facts, start/end values, exact changes,
    and directional classifications.
    
    Stress decreases -> improved, Stress increases -> worsening.
    Positive metrics increase -> improved, decrease -> declining.
    Zero change -> unchanged.
    """
    if data is None or data.empty:
        return {}

    # Take up to the last 7 records
    recent = data.tail(7).copy()
    days_count = len(recent)
    first_row = recent.iloc[0]
    last_row = recent.iloc[-1]

    from src.wellness_score import calculate_wellness_score

    def _extract_score(row: pd.Series) -> Optional[float]:
        sc = _valid_number(row.get("WellnessScore"))
        if sc is not None:
            return sc
        sc, _ = calculate_wellness_score(
            sleep=_valid_number(row.get("SleepHours")),
            mood=_valid_number(row.get("Mood")),
            stress=_valid_number(row.get("Stress")),
            energy=_valid_number(row.get("Energy")),
            exercise=_valid_number(row.get("ExerciseMinutes")),
            screen_time=_valid_number(row.get("ScreenTime")),
            productivity=_valid_number(row.get("Productivity")),
        )
        return sc

    start_score = _extract_score(first_row)
    end_score = _extract_score(last_row)
    wellness_change = round(end_score - start_score, 1) if (start_score is not None and end_score is not None) else 0.0
    wellness_direction = calculate_direction(wellness_change, "WellnessScore")

    tracked_metrics = [
        "Mood", "Energy", "Stress", "Productivity",
        "SleepHours", "ScreenTime", "ExerciseMinutes", "StudyHours"
    ]

    metrics_facts = {}
    for metric in tracked_metrics:
        if metric not in recent.columns:
            continue
        start_val = _valid_number(first_row.get(metric))
        end_val = _valid_number(last_row.get(metric))
        if start_val is None or end_val is None:
            continue

        change = round(end_val - start_val, 1)
        direction = calculate_direction(change, metric)
        avg_val = round(float(recent[metric].mean()), 1)

        metrics_facts[metric] = {
            "start": int(start_val) if metric in ["Mood", "Energy", "Stress", "Productivity", "ExerciseMinutes"] else start_val,
            "end": int(end_val) if metric in ["Mood", "Energy", "Stress", "Productivity", "ExerciseMinutes"] else end_val,
            "change": int(change) if metric in ["Mood", "Energy", "Stress", "Productivity", "ExerciseMinutes"] else change,
            "direction": direction,
            "average": avg_val,
        }

    # Identify largest and smallest absolute change among primary core 4
    core_metrics = ["Mood", "Energy", "Stress", "Productivity"]
    core_changes = {
        m: abs(metrics_facts[m]["change"])
        for m in core_metrics
        if m in metrics_facts
    }

    largest_change_metric = max(core_changes, key=core_changes.get) if core_changes else None
    smallest_change_metric = min(core_changes, key=core_changes.get) if core_changes else None

    return {
        "days_count": days_count,
        "wellness_start": int(round(start_score)) if start_score is not None else None,
        "wellness_end": int(round(end_score)) if end_score is not None else None,
        "wellness_change": int(round(wellness_change)) if wellness_change is not None else 0,
        "wellness_direction": wellness_direction,
        "metrics": metrics_facts,
        "largest_change_metric": largest_change_metric,
        "smallest_change_metric": smallest_change_metric,
    }


# =====================================================================
# STEP 5: FORECAST FACTS
# =====================================================================

def build_forecast_facts(
    current: Union[pd.Series, Dict[str, Any]],
    predicted: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate deterministic forecast differences and directions.
    
    For every predicted metric:
        current
        predicted
        change = predicted - current
        direction:
            improving / declining / worsening / unchanged
            
    Stress decreases -> improving, Stress increases -> worsening.
    Other metrics increase -> improving, decrease -> declining.
    """
    if not predicted:
        return {}

    if isinstance(current, pd.Series):
        curr_dict = current.to_dict()
    else:
        curr_dict = dict(current) if current else {}

    forecast_facts = {}
    metrics = ["Mood", "Energy", "Stress", "Productivity", "SleepHours", "ScreenTime", "ExerciseMinutes"]

    for metric in metrics:
        if metric not in predicted or predicted[metric] is None:
            continue

        pred_val = _valid_number(predicted[metric])
        curr_val = _valid_number(curr_dict.get(metric))
        if pred_val is None:
            continue

        if curr_val is not None:
            change = round(pred_val - curr_val, 1)
            raw_dir = calculate_direction(change, metric)
            # Map "improved" -> "improving" for forward-looking clarity
            if raw_dir == "improved":
                direction = "improving"
            elif raw_dir == "declining":
                direction = "declining"
            elif raw_dir == "worsening":
                direction = "worsening"
            else:
                direction = "unchanged"
        else:
            change = 0.0
            direction = "unchanged"

        forecast_facts[metric] = {
            "current": round(curr_val, 1) if curr_val is not None else None,
            "predicted": round(pred_val, 1),
            "change": round(change, 1),
            "direction": direction,
        }

    return forecast_facts


# =====================================================================
# STEP 6: DAILY NOTE FACTS
# =====================================================================

def classify_metric(metric: str, value: Any) -> str:
    """
    Deterministic rule-based classification based on standard LifeLens ranges.
    
    Ranges:
        Mood, Stress, Energy, Productivity: 1-10
        SleepHours: 0-24
        ScreenTime: 0-24
        ExerciseMinutes: 0-1440
        StudyHours: 0-24
    """
    num = _valid_number(value)
    if num is None:
        return "unrecorded"

    m = metric.lower()

    if m in ["mood", "energy", "productivity"]:
        if num >= 7:
            return "high"
        elif num >= 5:
            return "moderate"
        else:
            return "low"

    elif m == "stress":
        if num >= 7:
            return "high"
        elif num >= 4:
            return "moderate"
        else:
            return "low"

    elif "sleep" in m:
        if num >= 7.0:
            return "adequate"
        elif num >= 6.0:
            return "fair"
        else:
            return "short"

    elif "screen" in m:
        if num >= 6.0:
            return "high"
        elif num >= 3.0:
            return "moderate"
        else:
            return "low"

    elif "exercise" in m:
        if num >= 30:
            return "active"
        elif num > 0:
            return "light"
        else:
            return "none"

    elif "study" in m:
        if num >= 4.0:
            return "high"
        elif num >= 2.0:
            return "moderate"
        else:
            return "low"

    return "normal"


def build_daily_note_facts(
    note: Optional[str],
    latest: Union[pd.Series, Dict[str, Any]],
    wellness_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Build deterministic facts and semantic labels to support daily note analysis.
    """
    note_clean = str(note).strip() if note is not None else ""
    if note_clean.lower() in ["none", "nan", ""]:
        note_clean = ""

    if isinstance(latest, pd.Series):
        latest_dict = latest.to_dict()
    else:
        latest_dict = dict(latest) if latest else {}

    sleep = _valid_number(latest_dict.get("SleepHours"))
    screen = _valid_number(latest_dict.get("ScreenTime"))
    exercise = _valid_number(latest_dict.get("ExerciseMinutes"))
    study = _valid_number(latest_dict.get("StudyHours"))
    mood = _valid_number(latest_dict.get("Mood"))
    stress = _valid_number(latest_dict.get("Stress"))
    energy = _valid_number(latest_dict.get("Energy"))
    productivity = _valid_number(latest_dict.get("Productivity"))

    if wellness_score is None:
        from src.wellness_score import calculate_wellness_score
        wellness_score, _ = calculate_wellness_score(
            sleep=sleep, mood=mood, stress=stress, energy=energy,
            exercise=exercise, screen_time=screen, productivity=productivity
        )

    labels = {
        "mood_label": classify_metric("mood", mood),
        "stress_label": classify_metric("stress", stress),
        "energy_label": classify_metric("energy", energy),
        "productivity_label": classify_metric("productivity", productivity),
        "sleep_label": classify_metric("sleep", sleep),
        "screen_label": classify_metric("screen", screen),
        "exercise_label": classify_metric("exercise", exercise),
        "study_label": classify_metric("study", study),
    }

    return {
        "has_note": bool(note_clean),
        "note": note_clean,
        "wellness_score": int(round(wellness_score)) if wellness_score is not None else None,
        "metrics": {
            "Mood": mood,
            "Stress": stress,
            "Energy": energy,
            "Productivity": productivity,
            "SleepHours": sleep,
            "ScreenTime": screen,
            "ExerciseMinutes": exercise,
            "StudyHours": study,
        },
        "classifications": labels,
    }
