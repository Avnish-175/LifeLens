def generate_recommendations(data):

    if data.empty:
        return ["Start logging your daily habits to receive recommendations."]

    latest = data.iloc[-1]

    recommendations = []

    if latest["SleepHours"] < 7:
        recommendations.append(
            "💤 Aim for at least 7–8 hours of sleep tonight."
        )

    if latest["Stress"] >= 7:
        recommendations.append(
            "😌 Your stress is high. Try a short meditation or breathing exercise."
        )

    if latest["ScreenTime"] > 6:
        recommendations.append(
            "📱 Reduce screen time before bed to improve sleep quality."
        )

    if latest["ExerciseMinutes"] < 30:
        recommendations.append(
            "🏃 Try getting at least 30 minutes of physical activity today."
        )

    if latest["Mood"] <= 5:
        recommendations.append(
            "😊 Spend time doing something you enjoy or talking with a friend."
        )

    if latest["Energy"] <= 5:
        recommendations.append(
            "⚡ Consider taking short breaks and staying hydrated."
        )

    if latest["Productivity"] <= 5:
        recommendations.append(
            "🎯 Break today's work into smaller, manageable tasks."
        )

    if not recommendations:
        recommendations.append(
            "🌟 Excellent work! Keep maintaining your healthy routine."
        )

    return recommendations