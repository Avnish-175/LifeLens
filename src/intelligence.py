def get_biggest_discovery(data):
    """
    Analyse recent behaviour and return the most meaningful insight.
    """

    if data.empty:
        return "Start logging your daily habits to discover behavioural patterns."

    recent = data.tail(7)

    if len(recent) < 2:
        return "Keep logging for a few more days before meaningful behavioural patterns can be detected."

    sleep_change = recent["SleepHours"].iloc[-1] - recent["SleepHours"].iloc[0]
    stress_change = recent["Stress"].iloc[-1] - recent["Stress"].iloc[0]
    mood_change = recent["Mood"].iloc[-1] - recent["Mood"].iloc[0]
    productivity_change = (
        recent["Productivity"].iloc[-1]
        - recent["Productivity"].iloc[0]
    )
    screen_change = (
        recent["ScreenTime"].iloc[-1]
        - recent["ScreenTime"].iloc[0]
    )
    exercise_change = (
        recent["ExerciseMinutes"].iloc[-1]
        - recent["ExerciseMinutes"].iloc[0]
    )

    # --------------------------------------------------
    # Sleep seems to be driving stress
    # --------------------------------------------------

    if sleep_change <= -1 and stress_change >= 1:
        return (
            "Your sleep has gradually decreased over the last few days, "
            "while your stress has increased. "
            "Sleep appears to be the biggest factor affecting your wellbeing."
        )

    # --------------------------------------------------
    # Screen time affecting sleep
    # --------------------------------------------------

    if screen_change >= 1 and sleep_change <= -1:
        return (
            "Your screen time has been increasing while your sleep keeps "
            "declining. Reducing late-night screen usage may improve recovery."
        )

    # --------------------------------------------------
    # Exercise improving mood
    # --------------------------------------------------

    if exercise_change >= 20 and mood_change >= 1:
        return (
            "As your exercise became more consistent, your mood also improved. "
            "Your physical activity seems to be positively influencing your mental wellbeing."
        )

    # --------------------------------------------------
    # Better sleep improving productivity
    # --------------------------------------------------

    if sleep_change >= 1 and productivity_change >= 1:
        return (
            "Your improved sleep pattern is being reflected in higher productivity. "
            "Maintaining this routine could continue improving your performance."
        )

    # --------------------------------------------------
    # Mood declining
    # --------------------------------------------------

    if mood_change <= -2:
        return (
            "Your mood has gradually declined over the past week. "
            "It may be worth reviewing your sleep, stress, and daily workload."
        )

    # --------------------------------------------------
    # Default
    # --------------------------------------------------

    return (
        "No major behavioural shifts were detected this week. "
        "Your habits have remained fairly consistent."
    )

def get_todays_focus(data):
    """
    Return the single highest-impact action
    the user should take next.
    """

    if data.empty:
        return "Start logging your daily habits to receive personalized guidance."

    latest = data.iloc[-1]

    sleep = latest["SleepHours"]
    stress = latest["Stress"]
    screen = latest["ScreenTime"]
    exercise = latest["ExerciseMinutes"]
    productivity = latest["Productivity"]

    # Highest priority actions

    if sleep < 6:
        return "😴 Aim for at least 7 hours of sleep tonight."

    if stress >= 8:
        return "🧘 Take 15 minutes today to relax, meditate, or simply disconnect."

    if screen >= 8:
        return "📱 Reduce your screen time by at least 1 hour today."

    if exercise < 30:
        return "🚶 Try to complete at least 30 minutes of physical activity today."

    if productivity < 5:
        return "📚 Break your work into smaller tasks and focus on completing one at a time."

    return "🌱 Keep following your current routine. Your recent habits are well balanced."
