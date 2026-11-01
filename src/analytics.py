import pandas as pd


def get_summary(data):

    if data.empty:
        return {
            "total": 0,
            "avg_sleep": 0,
            "avg_mood": 0,
            "avg_stress": 0,
            "avg_screen": 0
        }

    return {
        "total": len(data),
        "avg_sleep": round(data["SleepHours"].mean(), 1),
        "avg_mood": round(data["Mood"].mean(), 1),
        "avg_stress": round(data["Stress"].mean(), 1),
        "avg_screen": round(data["ScreenTime"].mean(), 1)
    }