def _diff(s):
    try:
        valid = s.dropna()
        if len(valid) >= 2:
            return float(valid.iloc[-1] - valid.iloc[0])
    except Exception:
        pass
    return None


def get_biggest_discovery(data):
    """
    Analyse recent behaviour and return the most meaningful insight.
    """

    if data.empty:
        return "Start logging your daily habits to discover behavioural patterns."

    recent = data.tail(7)

    if len(recent) < 2:
        return "Keep logging for a few more days before meaningful behavioural patterns can be detected."

    sleep_change = _diff(recent["SleepHours"]) if "SleepHours" in recent.columns else None
    stress_change = _diff(recent["Stress"]) if "Stress" in recent.columns else None
    mood_change = _diff(recent["Mood"]) if "Mood" in recent.columns else None
    productivity_change = _diff(recent["Productivity"]) if "Productivity" in recent.columns else None
    screen_change = _diff(recent["ScreenTime"]) if "ScreenTime" in recent.columns else None
    exercise_change = _diff(recent["ExerciseMinutes"]) if "ExerciseMinutes" in recent.columns else None

    # --------------------------------------------------
    # Sleep seems to be driving stress
    # --------------------------------------------------

    if (
        sleep_change is not None and sleep_change <= -1
        and stress_change is not None and stress_change >= 1
    ):
        return (
            "Your sleep has gradually decreased over the last few days, "
            "while your stress has increased. "
            "Sleep appears to be the biggest factor affecting your wellbeing."
        )

    # --------------------------------------------------
    # Screen time affecting sleep
    # --------------------------------------------------

    if (
        screen_change is not None and screen_change >= 1
        and sleep_change is not None and sleep_change <= -1
    ):
        return (
            "Your screen time has been increasing while your sleep keeps "
            "declining. Reducing late-night screen usage may improve recovery."
        )

    # --------------------------------------------------
    # Exercise improving mood
    # --------------------------------------------------

    if (
        exercise_change is not None and exercise_change >= 20
        and mood_change is not None and mood_change >= 1
    ):
        return (
            "As your exercise became more consistent, your mood also improved. "
            "Your physical activity seems to be positively influencing your mental wellbeing."
        )

    # --------------------------------------------------
    # Better sleep improving productivity
    # --------------------------------------------------

    if (
        sleep_change is not None and sleep_change >= 1
        and productivity_change is not None and productivity_change >= 1
    ):
        return (
            "Your improved sleep pattern is being reflected in higher productivity. "
            "Maintaining this routine could continue improving your performance."
        )

    # --------------------------------------------------
    # Mood declining
    # --------------------------------------------------

    if mood_change is not None and mood_change <= -2:
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

    def _val(x):
        try:
            if x is not None and not (isinstance(x, float) and (x != x)):
                return float(x)
        except (ValueError, TypeError):
            pass
        return None

    sleep = _val(latest.get("SleepHours"))
    stress = _val(latest.get("Stress"))
    screen = _val(latest.get("ScreenTime"))
    exercise = _val(latest.get("ExerciseMinutes"))
    productivity = _val(latest.get("Productivity"))

    # Highest priority actions

    if sleep is not None and sleep < 6:
        return "😴 Aim for at least 7 hours of sleep tonight."

    if stress is not None and stress >= 8:
        return "🧘 Take 15 minutes today to relax, meditate, or simply disconnect."

    if screen is not None and screen >= 8:
        return "📱 Reduce your screen time by at least 1 hour today."

    if exercise is not None and exercise < 30:
        return "🚶 Try to complete at least 30 minutes of physical activity today."

    if productivity is not None and productivity < 5:
        return "📚 Break your work into smaller tasks and focus on completing one at a time."

    return "🌱 Keep following your current routine. Your recent habits are well balanced."
