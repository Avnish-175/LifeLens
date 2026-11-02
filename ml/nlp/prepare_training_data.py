import os
import sys
import random
import pandas as pd


# Add LifeLens project root to Python path
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, BASE_DIR)

from src.wellness_score import calculate_wellness_score


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_FILE = os.path.join(DATA_DIR, "ml_training_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "nlp_training_data.csv")


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

random.seed(42)

TARGET_ROWS = 3000


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def safe(value):
    if pd.isna(value):
        return None
    return float(value)


def build_report(score, category, row):
    sleep = safe(row["SleepHours"])
    mood = safe(row["Mood"])
    energy = safe(row["Energy"])
    stress = safe(row["Stress"])
    exercise = safe(row["ExerciseMinutes"])
    screen = safe(row["ScreenTime"])
    productivity = safe(row["Productivity"])

    # ---------------------------------------------------------
    # Calculate concern levels for each wellness factor.
    # Higher value = greater concern.
    # ---------------------------------------------------------

    concerns = {
        "mood": max(0, 10 - mood),
        "energy": max(0, 10 - energy),
        "stress": max(0, stress - 5),
        "exercise": max(0, 30 - exercise) / 3,
        "screen": max(0, screen - 4) * 1.5,
        "productivity": max(0, 10 - productivity),
    }

    # Sleep needs a special range because both too little
    # and too much sleep can be undesirable.
    if 7 <= sleep <= 9:
        sleep_concern = 0
    elif 6 <= sleep < 7 or 9 < sleep <= 10:
        sleep_concern = 2
    elif 5 <= sleep < 6 or 10 < sleep <= 11:
        sleep_concern = 5
    else:
        sleep_concern = 8

    concerns["sleep"] = sleep_concern

    # Rank the factors from biggest concern to smallest.
    ranked = sorted(
        concerns.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # ---------------------------------------------------------
    # Overall point
    # ---------------------------------------------------------

    overall = [
        f"Overall wellness is {category.lower()} with a score of {score}/100.",
        f"Your current wellness level is {category.lower()} at {score}/100.",
        f"The overall wellness picture is {category.lower()}, scoring {score}/100.",
    ]

    # ---------------------------------------------------------
    # Meaningful observations
    # ---------------------------------------------------------

    observations = {
        "mood": [
            f"Low mood at {mood:.1f}/10 is one of the biggest factors affecting today's wellness.",
            f"Your mood is relatively low at {mood:.1f}/10 and is an important area to improve.",
        ],

        "energy": [
            f"Energy is relatively low at {energy:.1f}/10 and may be limiting today's overall wellness.",
            f"Your energy level of {energy:.1f}/10 suggests that recovery and daily activity could be improved.",
        ],

        "stress": [
            f"Stress is relatively high at {stress:.1f}/10 and is putting pressure on today's wellness.",
            f"Managing the current stress level of {stress:.1f}/10 could improve the overall wellness picture.",
        ],

        "exercise": [
            f"Exercise is relatively low at {exercise:.0f} minutes today.",
            f"Only {exercise:.0f} minutes of physical activity were recorded, leaving room for improvement.",
        ],

        "screen": [
            f"Screen time is relatively high at {screen:.1f} hours.",
            f"Your screen exposure of {screen:.1f} hours could be reduced to support better daily balance.",
        ],

        "productivity": [
            f"Productivity is relatively low at {productivity:.1f}/10.",
            f"Your productivity level of {productivity:.1f}/10 suggests that daily focus could be improved.",
        ],

        "sleep": [
            f"Sleep duration is outside the typical healthy range at {sleep:.1f} hours.",
            f"Your {sleep:.1f} hours of sleep is outside the usual 7–9 hour range.",
        ],
    }

    # ---------------------------------------------------------
    # Choose the two most important concerns
    # ---------------------------------------------------------

    top_concerns = [
        factor
        for factor, value in ranked
        if value > 0
    ][:2]

    # ---------------------------------------------------------
    # Build relationship/action point
    # ---------------------------------------------------------

    if mood < 4 and stress >= 7:
        final_point = (
            "Lowering stress while supporting mood should be a major "
            "priority for improving overall wellness."
        )

    elif mood < 4 and energy < 5:
        final_point = (
            "Low mood and energy suggest that recovery and daily routine "
            "may need additional attention."
        )

    elif exercise < 10 and energy < 5:
        final_point = (
            "Increasing regular physical activity may help support "
            "energy and overall wellness."
        )

    elif sleep < 6 and energy < 6:
        final_point = (
            "Improving sleep consistency may help support better energy "
            "and recovery."
        )

    elif screen > 7 and sleep < 7:
        final_point = (
            "Reducing evening screen exposure may help support better "
            "sleep and daily recovery."
        )

    elif stress >= 7:
        final_point = (
            "Stress management should be one of the main priorities "
            "for improving the overall wellness profile."
        )

    elif exercise < 20:
        final_point = (
            "Increasing regular physical activity should be a useful "
            "priority."
        )

    else:
        final_point = (
            "Maintaining consistency across sleep, mood, energy and "
            "daily activity should help support overall wellness."
        )

    # ---------------------------------------------------------
    # Build the four-point report
    # ---------------------------------------------------------

    points = [
        random.choice(overall)
    ]

    for factor in top_concerns:
        sentence = random.choice(observations[factor])

        if sentence not in points:
            points.append(sentence)

        if len(points) == 3:
            break

    # Add a useful final action/relationship.
    if final_point not in points:
        points.append(final_point)

    # Safety fallback to guarantee exactly 4 points.
    while len(points) < 4:
        fallback = (
            "Maintaining balanced daily habits across sleep, activity "
            "and recovery should support better wellness."
        )

        if fallback not in points:
            points.append(fallback)
        else:
            points.append(
                "Consistent daily habits can help support long-term wellness."
            )

    return "\n".join(
        f"{index + 1}. {point}"
        for index, point in enumerate(points[:4])
    )


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("Loading LifeLens ML dataset...")

data = pd.read_csv(INPUT_FILE)

print(f"Source rows: {len(data)}")


required_columns = [
    "SleepHours",
    "ScreenTime",
    "ExerciseMinutes",
    "Mood",
    "Energy",
    "Stress",
    "Productivity",
]

missing = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ---------------------------------------------------------
# Remove incomplete rows
# ---------------------------------------------------------

data = data.dropna(
    subset=required_columns
).reset_index(drop=True)

print(f"Usable rows: {len(data)}")


# ---------------------------------------------------------
# Generate training examples
# ---------------------------------------------------------

training_rows = []

for _, row in data.iterrows():

    score, category = calculate_wellness_score(
        row["SleepHours"],
        row["Mood"],
        row["Stress"],
        row["Energy"],
        row["ExerciseMinutes"],
        row["ScreenTime"],
        row["Productivity"],
    )

    if score is None:
        continue

    prompt = f"""Generate a 4-point wellness report for this LifeLens user.

Wellness Score: {score}/100
Category: {category}

Sleep: {row["SleepHours"]:.1f} hours
Mood: {row["Mood"]:.1f}/10
Energy: {row["Energy"]:.1f}/10
Stress: {row["Stress"]:.1f}/10
Exercise: {row["ExerciseMinutes"]:.0f} minutes
Screen Time: {row["ScreenTime"]:.1f} hours
Productivity: {row["Productivity"]:.1f}/10
"""

    target = build_report(
        score,
        category,
        row
    )

    training_rows.append({
        "input_text": prompt,
        "target_text": target
    })


# ---------------------------------------------------------
# Duplicate/expand dataset if necessary
# ---------------------------------------------------------

if len(training_rows) < TARGET_ROWS:

    original = training_rows.copy()

    while len(training_rows) < TARGET_ROWS:
        training_rows.append(
            random.choice(original)
        )

training_rows = training_rows[:TARGET_ROWS]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output = pd.DataFrame(training_rows)

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("NLP TRAINING DATA CREATED")
print("=" * 60)

print(f"Examples: {len(output)}")
print(f"Saved to: {OUTPUT_FILE}")

print()
print("Example:")
print("-" * 60)
print(output.iloc[0]["input_text"])
print()
print(output.iloc[0]["target_text"])
print("-" * 60)