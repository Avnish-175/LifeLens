def generate_monthly_report(data):
    """
    AI Monthly Snapshot
    """

    if data.empty:
        return {}

    recent = data.tail(30)

    avg_sleep = round(recent["SleepHours"].mean(), 1)
    avg_mood = round(recent["Mood"].mean(), 1)
    avg_stress = round(recent["Stress"].mean(), 1)
    avg_energy = round(recent["Energy"].mean(), 1)
    avg_screen = round(recent["ScreenTime"].mean(), 1)
    avg_productivity = round(recent["Productivity"].mean(), 1)

    exercise_days = (recent["ExerciseMinutes"] > 0).sum()

    # ------------------------
    # Overall AI Score
    # ------------------------

    score = (
        avg_sleep * 10 +
        avg_mood * 10 +
        (10 - avg_stress) * 10 +
        avg_energy * 10 +
        avg_productivity * 10 +
        max(0, (10 - avg_screen)) * 5
    )

    overall_score = min(100, round(score / 5.5))

    # ------------------------
    # Biggest Win
    # ------------------------

    values = {
        "Sleep": avg_sleep,
        "Mood": avg_mood,
        "Energy": avg_energy,
        "Productivity": avg_productivity
    }

    biggest_win = max(values, key=values.get)

    # ------------------------
    # Biggest Challenge
    # ------------------------

    challenges = {
        "Stress": avg_stress,
        "Screen Time": avg_screen,
    }

    challenge = max(challenges, key=challenges.get)

    # ------------------------
    # Biggest Opportunity
    # ------------------------

    if avg_energy < 6:
        opportunity = "Improve your daily energy."

    elif avg_sleep < 7:
        opportunity = "Improve your sleep schedule."

    elif avg_stress > 6:
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