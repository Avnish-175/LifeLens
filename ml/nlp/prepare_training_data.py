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


DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "ml_training_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "nlp_training_data.csv")


# Reproducible generation with enough variation for fine-tuning.
random.seed(42)
TARGET_ROWS = 20000


REQUIRED_COLUMNS = [
    "SleepHours",
    "ScreenTime",
    "ExerciseMinutes",
    "Mood",
    "Energy",
    "Stress",
    "Productivity",
]


def safe(value):
    if pd.isna(value):
        return None
    return float(value)


def sleep_concern(sleep):
    if 7 <= sleep <= 9:
        return 0
    if 6 <= sleep < 7 or 9 < sleep <= 10:
        return 2
    if 5 <= sleep < 6 or 10 < sleep <= 11:
        return 5
    return 8


def build_report(score, category, row):
    """Create a varied four-section report matching reporter.py's format."""
    sleep = safe(row["SleepHours"])
    mood = safe(row["Mood"])
    energy = safe(row["Energy"])
    stress = safe(row["Stress"])
    exercise = safe(row["ExerciseMinutes"])
    screen = safe(row["ScreenTime"])
    productivity = safe(row["Productivity"])

    concerns = {
        "mood": max(0, 10 - mood),
        "energy": max(0, 10 - energy),
        "stress": max(0, stress - 5),
        "exercise": max(0, 30 - exercise) / 3,
        "screen": max(0, screen - 4) * 1.5,
        "productivity": max(0, 10 - productivity),
        "sleep": sleep_concern(sleep),
    }

    ranked = sorted(concerns.items(), key=lambda item: item[1], reverse=True)
    active = [factor for factor, value in ranked if value > 0]

    overall_templates = [
        "Overall wellness is {category} at {score}/100.",
        "Today's wellness picture is {category}, with a score of {score}/100.",
        "Your current LifeLens wellness level is {category} ({score}/100).",
        "The overall wellness score is {score}/100, placing today in the {category} range.",
        "LifeLens indicates a {category} wellness day with a score of {score}/100.",
        "Today's overall wellness stands at {score}/100 and is classified as {category}.",
    ]

    observations = {
        "mood": [
            "Mood is {mood:.1f}/10 and is one of the main areas influencing today's result.",
            "At {mood:.1f}/10, mood is a noticeable factor in the current wellness picture.",
            "Mood sits at {mood:.1f}/10, leaving room for improvement in today's balance.",
        ],
        "energy": [
            "Energy is {energy:.1f}/10 and may be limiting overall daily momentum.",
            "The current energy level is {energy:.1f}/10, making recovery and routine worth watching.",
            "Energy is recorded at {energy:.1f}/10 and is an important part of today's wellness profile.",
        ],
        "stress": [
            "Stress is {stress:.1f}/10 and is adding pressure to the current wellness profile.",
            "The stress level of {stress:.1f}/10 stands out as an area that could be managed better.",
            "At {stress:.1f}/10, stress is one of the stronger factors affecting today's result.",
        ],
        "exercise": [
            "Only {exercise:.0f} minutes of exercise were recorded today, leaving room for more activity.",
            "Physical activity is at {exercise:.0f} minutes, which is relatively low for the day.",
            "Today's {exercise:.0f} minutes of exercise suggest that movement could be increased.",
        ],
        "screen": [
            "Screen time is {screen:.1f} hours, which is relatively high for the day.",
            "The {screen:.1f} hours of screen time stand out as a possible balance issue.",
            "Screen exposure reached {screen:.1f} hours, so reducing unnecessary time may help daily balance.",
        ],
        "productivity": [
            "Productivity is {productivity:.1f}/10 and is an area where daily focus could improve.",
            "The productivity level of {productivity:.1f}/10 suggests that focus and routine deserve attention.",
            "Productivity sits at {productivity:.1f}/10 and contributes noticeably to today's wellness picture.",
        ],
        "sleep": [
            "Sleep is {sleep:.1f} hours, outside the usual 7–9 hour range.",
            "The recorded {sleep:.1f} hours of sleep suggests that sleep consistency could be improved.",
            "Sleep duration is {sleep:.1f} hours, making sleep routine an important part of today's picture.",
        ],
    }

    relationship_templates = {
        "mood_stress": [
            "High stress alongside lower mood makes stress management and emotional recovery a useful priority.",
            "The combination of stress and mood suggests focusing on a calmer routine and better recovery.",
        ],
        "mood_energy": [
            "Lower mood and energy together suggest that recovery and daily routine deserve extra attention.",
            "Supporting both mood and energy could help create a more balanced day.",
        ],
        "sleep_energy": [
            "Improving sleep consistency may support better energy and recovery.",
            "Sleep and energy together point toward making recovery a stronger part of the daily routine.",
        ],
        "screen_sleep": [
            "Reducing late-day screen exposure may support a more consistent sleep routine.",
            "High screen time combined with weaker sleep makes evening digital habits worth reviewing.",
        ],
        "exercise_energy": [
            "Adding manageable physical activity may help support energy and daily momentum.",
            "More regular movement could be a practical way to strengthen energy and routine.",
        ],
        "stress_productivity": [
            "Managing stress may also help create better conditions for focus and productivity.",
            "The stress-productivity combination suggests making a calmer, more focused routine a priority.",
        ],
    }

    priority_templates = [
        "Priority: focus first on {factor}, while keeping the rest of the routine consistent.",
        "A useful next step is to improve {factor} gradually rather than changing everything at once.",
        "For the next day, give extra attention to {factor} and monitor whether the overall balance improves.",
        "The clearest improvement opportunity is {factor}; small, consistent changes are more useful than drastic ones.",
    ]

    overall = random.choice(overall_templates).format(
        category=str(category).lower(), score=score
    )

    # Select the strongest meaningful factor, with a small amount of randomness
    # so equally-scored factors do not always produce identical reports.
    top_factor = active[0] if active else "daily consistency"
    if len(active) > 1 and random.random() < 0.30:
        top_factor = random.choice(active[: min(3, len(active))])

    key_factor = random.choice(observations[top_factor]).format(
        mood=mood,
        energy=energy,
        stress=stress,
        exercise=exercise,
        screen=screen,
        productivity=productivity,
        sleep=sleep,
    )

    # Generate a relationship/pattern statement when a meaningful pair exists.
    if mood < 4 and stress >= 7:
        pattern = random.choice(relationship_templates["mood_stress"])
    elif mood < 5 and energy < 5:
        pattern = random.choice(relationship_templates["mood_energy"])
    elif sleep < 6 and energy < 6:
        pattern = random.choice(relationship_templates["sleep_energy"])
    elif screen > 7 and sleep < 7:
        pattern = random.choice(relationship_templates["screen_sleep"])
    elif exercise < 10 and energy < 5:
        pattern = random.choice(relationship_templates["exercise_energy"])
    elif stress >= 7 and productivity < 5:
        pattern = random.choice(relationship_templates["stress_productivity"])
    elif active:
        pattern = random.choice([
            f"The strongest current signal is {active[0]}, so that factor is likely to have the biggest impact on near-term improvement.",
            f"The current pattern is mainly shaped by {active[0]}, with the other factors contributing to the overall balance.",
            f"Among today's measures, {active[0]} stands out most clearly as an improvement opportunity.",
        ])
    else:
        pattern = random.choice([
            "The main pattern is relatively balanced across today's tracked wellness measures.",
            "No single factor dominates the profile, suggesting that consistency is more important than a major correction.",
            "The current measures are fairly balanced, so maintaining the routine is a sensible approach.",
        ])

    priority = random.choice(priority_templates).format(factor=top_factor.replace("_", " "))

    return "\n".join([
        f"OVERALL: {overall}",
        f"KEY FACTOR: {key_factor}",
        f"PATTERN: {pattern}",
        f"PRIORITY: {priority}",
    ])


def build_prompt(row, score, category):
    return f"""Generate a concise four-section LifeLens wellness report.

Wellness Score: {score}/100
Category: {category}
Sleep: {row['SleepHours']:.1f} hours
Mood: {row['Mood']:.1f}/10
Energy: {row['Energy']:.1f}/10
Stress: {row['Stress']:.1f}/10
Exercise: {row['ExerciseMinutes']:.0f} minutes
Screen Time: {row['ScreenTime']:.1f} hours
Productivity: {row['Productivity']:.1f}/10

Return exactly these sections:
OVERALL:
KEY FACTOR:
PATTERN:
PRIORITY:
"""


print("Loading LifeLens ML dataset...")
data = pd.read_csv(INPUT_FILE)
print(f"Source rows: {len(data)}")

missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

data = data.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
if data.empty:
    raise ValueError("No usable rows remain after validation.")

print(f"Usable rows: {len(data)}")

training_rows = []

# Sampling with replacement is intentional when the source is smaller than
# the target. Text is regenerated every time, so examples are not duplicated
# verbatim as in the previous pipeline.
for index in range(TARGET_ROWS):
    row = data.iloc[index % len(data)].copy()

    # Tiny bounded perturbations increase textual/scenario diversity without
    # moving values outside the LifeLens feature ranges.
    if index >= len(data):
        row["SleepHours"] = min(24.0, max(0.0, float(row["SleepHours"]) + random.uniform(-0.25, 0.25)))
        row["ScreenTime"] = min(24.0, max(0.0, float(row["ScreenTime"]) + random.uniform(-0.30, 0.30)))
        row["ExerciseMinutes"] = min(1440.0, max(0.0, float(row["ExerciseMinutes"]) + random.uniform(-5, 5)))
        for column in ["Mood", "Energy", "Stress", "Productivity"]:
            row[column] = min(10.0, max(1.0, float(row[column]) + random.uniform(-0.20, 0.20)))

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

    training_rows.append({
        "input_text": build_prompt(row, score, category),
        "target_text": build_report(score, category, row),
    })

output = pd.DataFrame(training_rows)
if output.empty:
    raise ValueError("No NLP training examples were generated.")

output = output.drop_duplicates(subset=["input_text", "target_text"]).reset_index(drop=True)

# If duplicate removal made the dataset smaller, continue generating fresh
# variants until the requested size is reached.
while len(output) < TARGET_ROWS:
    row = data.sample(1, random_state=random.randint(0, 2_000_000_000)).iloc[0].copy()
    row["SleepHours"] = min(24.0, max(0.0, float(row["SleepHours"]) + random.uniform(-0.35, 0.35)))
    row["ScreenTime"] = min(24.0, max(0.0, float(row["ScreenTime"]) + random.uniform(-0.40, 0.40)))
    row["ExerciseMinutes"] = min(1440.0, max(0.0, float(row["ExerciseMinutes"]) + random.uniform(-8, 8)))
    for column in ["Mood", "Energy", "Stress", "Productivity"]:
        row[column] = min(10.0, max(1.0, float(row[column]) + random.uniform(-0.25, 0.25)))

    score, category = calculate_wellness_score(
        row["SleepHours"], row["Mood"], row["Stress"], row["Energy"],
        row["ExerciseMinutes"], row["ScreenTime"], row["Productivity"]
    )
    candidate = {
        "input_text": build_prompt(row, score, category),
        "target_text": build_report(score, category, row),
    }
    if not ((output["input_text"] == candidate["input_text"]) & (output["target_text"] == candidate["target_text"])).any():
        output = pd.concat([output, pd.DataFrame([candidate])], ignore_index=True)

output = output.head(TARGET_ROWS)
output.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 60)
print("NLP TRAINING DATA CREATED")
print("=" * 60)
print(f"Examples: {len(output)}")
print(f"Unique inputs: {output['input_text'].nunique()}")
print(f"Unique targets: {output['target_text'].nunique()}")
print(f"Saved to: {OUTPUT_FILE}")
print()
print("Example:")
print("-" * 60)
print(output.iloc[0]["input_text"])
print()
print(output.iloc[0]["target_text"])
print("-" * 60)
