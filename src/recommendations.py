def generate_recommendations(data):

    if data.empty:
        return ["Start logging your daily habits to receive recommendations."]

    latest = data.iloc[-1]

    recommendations = []

    sleep = latest.get("SleepHours")
    stress = latest.get("Stress")
    screen = latest.get("ScreenTime")
    exercise = latest.get("ExerciseMinutes")
    mood = latest.get("Mood")
    energy = latest.get("Energy")
    productivity = latest.get("Productivity")

    def _val(x):
        try:
            if x is not None and not (isinstance(x, float) and (x != x)):
                return float(x)
        except (ValueError, TypeError):
            pass
        return None

    s = _val(sleep)
    st = _val(stress)
    sc = _val(screen)
    ex = _val(exercise)
    m = _val(mood)
    en = _val(energy)
    pr = _val(productivity)

    if s is not None and s < 7:
        recommendations.append(
            "💤 Aim for at least 7–8 hours of sleep tonight."
        )

    if st is not None and st >= 7:
        recommendations.append(
            "😌 Your stress is high. Try a short meditation or breathing exercise."
        )

    if sc is not None and sc > 6:
        recommendations.append(
            "📱 Reduce screen time before bed to improve sleep quality."
        )

    if ex is not None and ex < 30:
        recommendations.append(
            "🏃 Try getting at least 30 minutes of physical activity today."
        )

    if m is not None and m <= 5:
        recommendations.append(
            "😊 Spend time doing something you enjoy or talking with a friend."
        )

    if en is not None and en <= 5:
        recommendations.append(
            "⚡ Consider taking short breaks and staying hydrated."
        )

    if pr is not None and pr <= 5:
        recommendations.append(
            "🎯 Break today's work into smaller, manageable tasks."
        )

    if not recommendations:
        recommendations.append(
            "🌟 Excellent work! Keep maintaining your healthy routine."
        )

    return recommendations
