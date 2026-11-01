import numpy as np
import pandas as pd


def generate_insights(data):
    """
    Generate intelligent behavioural insights
    by analysing relationships between habits.
    """

    if data.empty:
        return [
            {
                "title": "Getting Started",
                "icon": "🌱",
                "status": "info",
                "message": "Start logging daily check-ins to receive personalised behavioural insights."
            }
        ]

    recent = data.tail(7)
    latest = recent.iloc[-1]

    insights = []

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
    mood = _val(latest.get("Mood"))
    energy = _val(latest.get("Energy"))
    productivity = _val(latest.get("Productivity"))

    # ---------------------------------
    # Sleep → Stress
    # ---------------------------------

    if sleep is not None and stress is not None and sleep < 6 and stress >= 7:
        insights.append({
            "title": "Sleep & Stress",
            "icon": "😴",
            "status": "danger",
            "message": "Poor sleep appears to be increasing your stress. Improving your sleep schedule could have the biggest positive impact on your wellbeing."
        })

    # ---------------------------------
    # Screen Time → Sleep
    # ---------------------------------

    if screen is not None and sleep is not None and screen >= 8 and sleep < 7:
        insights.append({
            "title": "Screen Time",
            "icon": "📱",
            "status": "warning",
            "message": "High screen time may be reducing your sleep quality, affecting both focus and recovery."
        })

    # ---------------------------------
    # Exercise → Mood
    # ---------------------------------

    if exercise is not None and mood is not None and exercise >= 30 and mood >= 7:
        insights.append({
            "title": "Exercise",
            "icon": "💪",
            "status": "success",
            "message": "Your physical activity is supporting a positive mood and overall wellbeing."
        })

    elif exercise is not None and exercise < 20:
        insights.append({
            "title": "Exercise",
            "icon": "🚶",
            "status": "warning",
            "message": "Increasing your daily physical activity could improve both your mood and energy."
        })

    # ---------------------------------
    # Sleep → Productivity
    # ---------------------------------

    if sleep is not None and productivity is not None and sleep < 6 and productivity <= 5:
        insights.append({
            "title": "Productivity",
            "icon": "🎯",
            "status": "warning",
            "message": "Lower productivity may be linked to insufficient sleep rather than a lack of effort."
        })

    elif productivity is not None and productivity >= 8:
        insights.append({
            "title": "Productivity",
            "icon": "🚀",
            "status": "success",
            "message": "Your current routine is supporting focused and productive work."
        })

    # ---------------------------------
    # Energy
    # ---------------------------------

    if energy is not None and energy <= 4:
        insights.append({
            "title": "Energy",
            "icon": "🔋",
            "status": "warning",
            "message": "Low energy may be linked to poor recovery, high stress, or insufficient activity."
        })

    elif energy is not None and energy >= 8:
        insights.append({
            "title": "Energy",
            "icon": "⚡",
            "status": "success",
            "message": "Your recent habits are supporting excellent daily energy levels."
        })

    # ---------------------------------
    # Balanced Lifestyle
    # ---------------------------------

    if not insights:
        insights.append({
            "title": "Balanced Lifestyle",
            "icon": "🌱",
            "status": "success",
            "message": "Your recent habits appear balanced. Focus on maintaining consistency rather than making major changes."
        })

    return insights


def _diff_series(s):
    try:
        valid = s.dropna()
        if len(valid) >= 2:
            return float(valid.iloc[-1] - valid.iloc[0])
    except Exception:
        pass
    return None


def generate_trend_analysis(data):
    """
    Generate AI-powered behavioural trend analysis
    from the most recent check-ins.
    """

    if len(data) < 2:
        return {
            "positive": "Keep logging your daily habits to unlock personalised trend analysis.",
            "watch": "More data is needed before meaningful trends can be identified.",
            "risk": "No significant behavioural risks detected yet.",
            "recommendation": "Continue tracking your wellbeing consistently."
        }

    recent = data.tail(7)

    sleep_change = _diff_series(recent["SleepHours"]) if "SleepHours" in recent.columns else None
    mood_change = _diff_series(recent["Mood"]) if "Mood" in recent.columns else None
    stress_change = _diff_series(recent["Stress"]) if "Stress" in recent.columns else None
    screen_change = _diff_series(recent["ScreenTime"]) if "ScreenTime" in recent.columns else None

    # -------------------------
    # Positive
    # -------------------------

    if mood_change is not None and mood_change >= 1:
        positive = "Your mood has improved over the past week."
    elif sleep_change is not None and sleep_change >= 1:
        positive = "Your sleep duration has improved recently."
    elif stress_change is not None and stress_change <= -1:
        positive = "Your stress levels have reduced over recent check-ins."
    else:
        positive = "Your wellbeing has remained fairly consistent this week."

    # -------------------------
    # Watch
    # -------------------------

    if sleep_change is not None and sleep_change <= -1:
        watch = "Your sleep duration has gradually declined."
    elif mood_change is not None and mood_change <= -1:
        watch = "Your mood has shown a slight downward trend."
    elif stress_change is not None and stress_change >= 1:
        watch = "Stress levels are beginning to increase."
    else:
        watch = "No major behavioural changes require close monitoring."

    # -------------------------
    # Biggest Risk
    # -------------------------

    if screen_change is not None and screen_change >= 2:
        risk = "Screen time has increased noticeably and may affect recovery."
    elif stress_change is not None and stress_change >= 2:
        risk = "Stress has risen significantly over recent days."
    elif sleep_change is not None and sleep_change <= -2:
        risk = "Reduced sleep may begin affecting your energy and productivity."
    else:
        risk = "No major health risks are currently detected."

    # -------------------------
    # AI Recommendation
    # -------------------------

    if sleep_change is not None and sleep_change <= -1:
        recommendation = "Aim for 7–8 hours of sleep over the next few days."
    elif screen_change is not None and screen_change >= 2:
        recommendation = "Reduce evening screen time by about one hour."
    elif stress_change is not None and stress_change >= 1:
        recommendation = "Schedule short breaks during the day to manage stress."
    elif mood_change is not None and mood_change <= -1:
        recommendation = "Spend time on activities that help improve your mood."
    else:
        recommendation = "Maintain your current routine and focus on consistency."

    return {
        "positive": positive,
        "watch": watch,
        "risk": risk,
        "recommendation": recommendation
    }


def generate_correlation_insights(correlation_matrix):
    """
    Return the strongest positive and negative relationships.
    Handle empty matrices and missing correlations safely.
    """
    if correlation_matrix is None or correlation_matrix.empty:
        return None

    corr = correlation_matrix.copy(deep=True)
    values = corr.to_numpy(dtype=float, copy=True)

    # Ignore self-correlations.
    np.fill_diagonal(values, np.nan)

    # Find only finite, non-diagonal correlations.
    valid_mask = np.isfinite(values)

    if not valid_mask.any():
        return None

    # Positive relationships: valid values greater than zero.
    positive_mask = valid_mask & (values > 0)

    # Negative relationships: valid values less than zero.
    negative_mask = valid_mask & (values < 0)

    def strength_label(value):
        abs_v = abs(value)
        if abs_v >= 0.90:
            return "Very Strong"
        elif abs_v >= 0.70:
            return "Strong"
        elif abs_v >= 0.50:
            return "Moderate"
        else:
            return "Weak"

    def relationship_details(row, col):
        value = round(float(values[row, col]), 2)
        abs_val = abs(value)

        return {
            "feature1": str(corr.index[row]),
            "feature2": str(corr.columns[col]),
            "value": value,
            "percent": int(round(abs_val * 100)),
            "strength": strength_label(value),
        }

    positive = None
    negative = None

    if positive_mask.any():
        positive_values = np.where(positive_mask, values, -np.inf)
        positive_idx = np.argmax(positive_values)
        pos_row, pos_col = np.unravel_index(positive_idx, values.shape)
        positive = relationship_details(pos_row, pos_col)

    if negative_mask.any():
        negative_values = np.where(negative_mask, values, np.inf)
        negative_idx = np.argmin(negative_values)
        neg_row, neg_col = np.unravel_index(negative_idx, values.shape)
        negative = relationship_details(neg_row, neg_col)

    return {
        "positive": positive,
        "negative": negative,
    }


def generate_relationship_explanation(feature1, feature2, value):
    """
    Generates a human-friendly AI explanation for a relationship.
    """

    names = {
        "SleepHours": "😴 Sleep",
        "Mood": "😊 Mood",
        "Stress": "😌 Stress",
        "Energy": "⚡ Energy",
        "ExerciseMinutes": "🏃 Exercise",
        "ScreenTime": "💻 Screen Time",
        "StudyHours": "📚 Study Hours",
        "Productivity": "🚀 Productivity"
    }

    f1 = names.get(feature1, feature1)
    f2 = names.get(feature2, feature2)

    if value > 0:
        return (
            f"Your recent check-ins suggest that <strong>{f1}</strong> and "
            f"<strong>{f2}</strong> tend to improve together. "
            "When one improves, the other usually improves as well."
        )

    return (
        f"Your recent check-ins suggest an inverse relationship between "
        f"<strong>{f1}</strong> and <strong>{f2}</strong>. "
        "As one increases, the other usually decreases."
    )
