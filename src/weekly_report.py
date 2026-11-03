def calculate_weekly_facts(data):
    """
    Calculate factual weekly statistics for AI explanation.

    Python is the source of truth.
    The LLM should only explain these calculated facts.
    """

    if data is None or data.empty:
        return {}

    # Use only available records
    data = data.copy()

    # Remove rows where all tracked metrics are missing
    metrics = [
        "SleepHours",
        "ScreenTime",
        "ExerciseMinutes",
        "Mood",
        "Energy",
        "Stress",
        "Productivity",
    ]

    available_metrics = [m for m in metrics if m in data.columns]

    if not available_metrics:
        return {}

    data = data.dropna(
        subset=available_metrics,
        how="all"
    )

    if data.empty:
        return {}

    # Latest 7 records = current period
    current = data.tail(7).copy()

    # Previous 7 records only exist when we have 14+ records
    previous = None

    if len(data) >= 14:
        previous = data.iloc[-14:-7].copy()

    facts = {
        "days_available": len(data),
        "current_period_days": len(current),
        "previous_period_available": previous is not None,
        "metrics": {},
    }

    def calculate_trend(change):
        if change is None:
            return "insufficient_data"

        if change > 0.5:
            return "increasing"
        elif change < -0.5:
            return "decreasing"
        else:
            return "stable"

    for metric in available_metrics:

        current_values = current[metric].dropna()

        if current_values.empty:
            continue

        current_average = round(
            float(current_values.mean()),
            2
        )

        previous_average = None
        change = None

        if previous is not None:
            previous_values = previous[metric].dropna()

            if not previous_values.empty:
                previous_average = round(
                    float(previous_values.mean()),
                    2
                )

                change = round(
                    current_average - previous_average,
                    2
                )

        metric_facts = {
            "current_average": current_average,
            "previous_average": previous_average,
            "change": change,
            "trend": calculate_trend(change),
        }

        # Additional exercise-specific fact
        if metric == "ExerciseMinutes":
            metric_facts["exercise_days"] = int(
                (current_values > 0).sum()
            )

        facts["metrics"][metric] = metric_facts

    return facts


def generate_weekly_report(data):

    if data.empty:
        return {}

    avg_sleep = data["SleepHours"].mean()
    avg_mood = data["Mood"].mean()
    avg_stress = data["Stress"].mean()
    avg_screen = data["ScreenTime"].mean()
    avg_exercise = data["ExerciseMinutes"].mean()
    avg_productivity = data["Productivity"].mean()
    avg_energy = data["Energy"].mean()

    # ------------------------
    # AI Headline
    # ------------------------

    if avg_sleep >= 7 and avg_mood >= 7 and avg_stress <= 4:
        headline = "🌟 You built a healthy routine this week."

    elif avg_sleep >= 6 and avg_mood >= 6:
        headline = "🙂 Your habits are becoming more consistent."

    else:
        headline = "💪 Every healthy habit this week is a step forward."

    # ------------------------
    # AI Reflection
    # ------------------------

    reflection = (
    f"You averaged {avg_sleep:.1f} hours of sleep, "
    f"maintained a {avg_mood:.1f}/10 mood, "
    f"and recorded {avg_stress:.1f}/10 stress this week. "
    "Your recent behaviour shows encouraging consistency with opportunities "
    "to improve through small daily habits."
)

   # ------------------------
# Weekly Badge
# ------------------------

    if avg_sleep >= 7:
        badge = "😴 Sleep Master"
        badge_description = "You consistently maintained healthy sleep throughout the week."

    elif avg_productivity >= 8:
        badge = "🚀 Productivity Hero"
        badge_description = "Excellent productivity made this your strongest habit."

    elif avg_exercise >= 30:
        badge = "🏃 Active Achiever"
        badge_description = "You stayed physically active on most days."

    elif avg_stress <= 3:
        badge = "🧘 Calm Mind"
        badge_description = "You successfully maintained a calm and balanced week."

    else:
        badge = "🌟 Consistency Champion"
        badge_description = "Your overall consistency is improving every week."


# ------------------------
# AI Next Week Mission
# ------------------------

    if avg_energy < 6:
        mission = "Boost Your Energy"
        mission_reason = (
        "Energy is currently your biggest opportunity for improving overall wellbeing."
    )

        mission_actions = [
        "Sleep before 11 PM",
        "Walk for at least 20 minutes daily",
        "Reduce screen time after 10 PM"
    ]

    elif avg_sleep < 7:
        mission = "Improve Your Sleep"

        mission_reason = (
        "Improving sleep will positively influence your mood, energy and productivity."
    )

        mission_actions = [
        "Maintain a fixed bedtime",
        "Avoid caffeine late in the evening",
        "Keep your phone away before sleeping"
    ]

    elif avg_stress > 6:
        mission = "Reduce Your Stress"

        mission_reason = (
        "Lower stress can improve both mood and productivity."
    )

        mission_actions = [
        "Practice deep breathing",
        "Take short work breaks",
        "Exercise for at least 20 minutes"
    ]

    else:
        mission = "Maintain Your Healthy Routine"

        mission_reason = (
        "Your habits are becoming consistent. Focus on maintaining your current routine."
    )

        mission_actions = [
        "Continue daily check-ins",
        "Stay physically active",
        "Protect your sleep schedule"
    ]
        # ------------------------
# Weekly Status Cards
# ------------------------

    sleep_status = "Good" if avg_sleep >= 7 else "😴 Needs Work"

    mood_status = "Positive" if avg_mood >= 7 else "😊 Stable"

    stress_status = "Calm" if avg_stress <= 4 else "😌 Elevated"

    energy_status = "High" if avg_energy >= 6 else "⚡ Needs Boost"
    return {

    "headline": headline,

    "reflection": reflection,

    "sleep_status": sleep_status,

    "mood_status": mood_status,

    "stress_status": stress_status,

    "energy_status": energy_status,

    "badge": badge,

    "badge_description": badge_description,

    "mission": mission,

    "mission_reason": mission_reason,

    "mission_actions": mission_actions

}