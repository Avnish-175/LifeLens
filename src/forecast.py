import pandas as pd


def predict_next_day(data):
    """
    Very simple trend prediction based on
    the average of the last 3 records.
    """

    if len(data) < 3:
        return {}

    recent = data.tail(3)

    prediction = {
        "SleepHours": round(recent["SleepHours"].mean(), 1),
        "Mood": round(recent["Mood"].mean(), 1),
        "Stress": round(recent["Stress"].mean(), 1),
        "Energy": round(recent["Energy"].mean(), 1),
        "ScreenTime": round(recent["ScreenTime"].mean(), 1),
        "ExerciseMinutes": round(recent["ExerciseMinutes"].mean(), 1),
        "Productivity": round(recent["Productivity"].mean(), 1),
    }

    return prediction

def generate_forecast_summary(prediction):

    """
    Generate an overall AI forecast title.
    """

    if not prediction:
        return "⚪ Insufficient data"

    sleep = prediction["SleepHours"]
    mood = prediction["Mood"]
    stress = prediction["Stress"]
    energy = prediction["Energy"]

    if sleep >= 7 and mood >= 7 and stress <= 4:
        return "🟢 Tomorrow looks Excellent"

    elif sleep >= 6.5 and mood >= 6 and stress <= 6:
        return "🟡 Tomorrow looks Stable"

    else:
        return "🔴 Recovery Day Expected"

def generate_forecast_explanation(prediction):

    """
    Explain WHY tomorrow is predicted this way.
    """

    if not prediction:
        return []

    explanation = []

    sleep = prediction["SleepHours"]
    mood = prediction["Mood"]
    stress = prediction["Stress"]
    energy = prediction["Energy"]
    screen = prediction["ScreenTime"]

    # Sleep

    if sleep >= 7:
        explanation.append(
            "😴 Your recent sleep pattern supports healthy recovery."
        )
    else:
        explanation.append(
            "😴 Increasing tonight's sleep could improve tomorrow's wellbeing."
        )

    # Mood

    if mood >= 7:
        explanation.append(
            "😊 Your mood is expected to remain positive."
        )
    else:
        explanation.append(
            "😊 Mood may improve with better rest and physical activity."
        )

    # Stress

    if stress <= 4:
        explanation.append(
            "😌 Stress levels are predicted to remain manageable."
        )
    else:
        explanation.append(
            "😌 Managing workload and taking breaks may reduce tomorrow's stress."
        )

    # Screen Time

    if screen >= 7:
        explanation.append(
            "📱 Reducing evening screen time could improve recovery."
        )

    return explanation

def generate_ai_forecast_insight(prediction):
    """
    Generate a natural-language AI insight
    explaining tomorrow's forecast.
    """

    if not prediction:
        return (
            "Continue logging your daily habits to unlock personalised AI forecasting."
        )

    sleep = prediction["SleepHours"]
    mood = prediction["Mood"]
    stress = prediction["Stress"]
    energy = prediction["Energy"]
    screen = prediction["ScreenTime"]
    productivity = prediction["Productivity"]

    # Excellent forecast
    if sleep >= 7 and mood >= 7 and stress <= 4:

        return (
            "Your recent habits indicate a well-balanced routine. "
            "Consistent sleep, positive mood and manageable stress are expected "
            "to support another productive and healthy day tomorrow."
        )

    # Screen time is biggest issue
    if screen >= 7:

        return (
            "Your overall wellness remains stable, but higher-than-ideal screen "
            "time may reduce tomorrow's recovery and energy. Reducing evening "
            "screen exposure could noticeably improve your forecast."
        )

    # Sleep issue
    if sleep < 6.5:

        return (
            "Sleep appears to be the strongest factor influencing tomorrow's "
            "forecast. Improving tonight's sleep duration is likely to produce "
            "the biggest improvement in mood, recovery and productivity."
        )

    # Stress issue
    if stress >= 6:

        return (
            "Stress is expected to have the greatest impact on tomorrow's wellbeing. "
            "Scheduling short recovery breaks and maintaining physical activity "
            "may help improve your overall wellness."
        )

    # Low mood
    if mood < 6:

        return (
            "Tomorrow's mood may remain slightly lower than usual. "
            "Maintaining healthy sleep, movement and social interaction could "
            "help improve emotional wellbeing."
        )

    # Default

    return (
        "Your recent behavioural patterns are consistent and suggest a stable "
        "wellbeing forecast. Continue your current routine while making small "
        "incremental improvements."
    )