import pandas as pd


def generate_monthly_report(data):
    """
    AI Monthly Snapshot
    """

    if data.empty:
        return {}

    recent = data.tail(30)

    avg_sleep = round(recent["SleepHours"].mean(), 1) if recent["SleepHours"].notna().any() else 0.0
    avg_mood = round(recent["Mood"].mean(), 1) if recent["Mood"].notna().any() else None
    avg_stress = round(recent["Stress"].mean(), 1) if recent["Stress"].notna().any() else None
    avg_energy = round(recent["Energy"].mean(), 1) if recent["Energy"].notna().any() else None
    avg_screen = round(recent["ScreenTime"].mean(), 1) if recent["ScreenTime"].notna().any() else 0.0
    avg_productivity = round(recent["Productivity"].mean(), 1) if recent["Productivity"].notna().any() else None

    exercise_days = int((recent["ExerciseMinutes"] > 0).sum())

    # ------------------------
    # Overall AI Score (proportional based on available factors)
    # ------------------------

    score_components = []
    max_components = 0.0

    if avg_sleep is not None and not pd.isna(avg_sleep):
        score_components.append(avg_sleep * 10)
        max_components += 100.0

    if avg_mood is not None and not pd.isna(avg_mood):
        score_components.append(avg_mood * 10)
        max_components += 100.0

    if avg_stress is not None and not pd.isna(avg_stress):
        score_components.append((10 - avg_stress) * 10)
        max_components += 100.0

    if avg_energy is not None and not pd.isna(avg_energy):
        score_components.append(avg_energy * 10)
        max_components += 100.0

    if avg_productivity is not None and not pd.isna(avg_productivity):
        score_components.append(avg_productivity * 10)
        max_components += 100.0

    if avg_screen is not None and not pd.isna(avg_screen):
        score_components.append(max(0, (10 - avg_screen)) * 5)
        max_components += 50.0

    if max_components > 0:
        overall_score = min(100, round((sum(score_components) / max_components) * 100))
    else:
        overall_score = 0

    # ------------------------
    # Biggest Win (from available factors)
    # ------------------------

    values = {
        "Sleep": avg_sleep,
        "Mood": avg_mood,
        "Energy": avg_energy,
        "Productivity": avg_productivity
    }
    available_values = {k: v for k, v in values.items() if v is not None and not pd.isna(v)}

    if available_values:
        biggest_win = max(available_values, key=available_values.get)
    else:
        biggest_win = "Sleep"

    # ------------------------
    # Biggest Challenge
    # ------------------------

    challenges = {
        "Stress": avg_stress,
        "Screen Time": avg_screen,
    }
    available_challenges = {k: v for k, v in challenges.items() if v is not None and not pd.isna(v)}

    if available_challenges:
        challenge = max(available_challenges, key=available_challenges.get)
    else:
        challenge = "Screen Time"

    # ------------------------
    # Biggest Opportunity
    # ------------------------

    if avg_energy is not None and not pd.isna(avg_energy) and avg_energy < 6:
        opportunity = "Improve your daily energy."

    elif avg_sleep is not None and not pd.isna(avg_sleep) and avg_sleep < 7:
        opportunity = "Improve your sleep schedule."

    elif avg_stress is not None and not pd.isna(avg_stress) and avg_stress > 6:
        opportunity = "Reduce daily stress."

    else:
        opportunity = "Maintain your healthy routine."

    return {

        "sleep": avg_sleep,
        "mood": avg_mood,
        "stress": avg_stress,
        "energy": avg_energy,
        "screen": avg_screen,
        "productivity": avg_productivity,

        "exercise_days": exercise_days,

        "overall_score": overall_score,

        "biggest_win": biggest_win,
        "challenge": challenge,
        "opportunity": opportunity

    }
