def analyze_patterns(data):
    """
    Analyze recent behaviour trends and return
    human-readable pattern summaries.
    """

    if data.empty:
        return ["Start logging daily check-ins to detect patterns."]

    # Analyze the last 7 entries (or fewer if not available)
    recent = data.tail(7)

    patterns = []

    # ---------- Sleep ----------
    if len(recent) >= 2:
        if recent["SleepHours"].iloc[-1] > recent["SleepHours"].iloc[0]:
            patterns.append(
                "😴 Your sleep has improved over the recent days."
            )
        elif recent["SleepHours"].iloc[-1] < recent["SleepHours"].iloc[0]:
            patterns.append(
                "😴 Your sleep has been decreasing recently. Try maintaining a consistent bedtime."
            )

    # ---------- Mood ----------
    if len(recent) >= 2:
        if recent["Mood"].iloc[-1] > recent["Mood"].iloc[0]:
            patterns.append(
                "😊 Your mood has been improving recently."
            )
        elif recent["Mood"].iloc[-1] < recent["Mood"].iloc[0]:
            patterns.append(
                "😔 Your mood has been declining. Consider taking time to recharge."
            )

    # ---------- Stress ----------
    if len(recent) >= 2:
        if recent["Stress"].iloc[-1] > recent["Stress"].iloc[0]:
            patterns.append(
                "⚠️ Your stress has increased over the last few days."
            )
        elif recent["Stress"].iloc[-1] < recent["Stress"].iloc[0]:
            patterns.append(
                "🌿 Great! Your stress has been reducing recently."
            )

    # ---------- Screen Time ----------
    if len(recent) >= 2:
        if recent["ScreenTime"].iloc[-1] > recent["ScreenTime"].iloc[0]:
            patterns.append(
                "📱 Your screen time is increasing. Consider taking regular digital breaks."
            )

    # ---------- Exercise ----------
    if recent["ExerciseMinutes"].mean() >= 30:
        patterns.append(
            "💪 You've maintained a healthy exercise routine this week."
        )
    else:
        patterns.append(
            "🏃 Try being more consistent with physical activity."
        )

    if not patterns:
        patterns.append(
            "🌟 Your habits have remained fairly consistent recently."
        )

    return patterns