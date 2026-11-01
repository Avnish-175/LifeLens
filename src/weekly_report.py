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