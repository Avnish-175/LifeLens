import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.wellness_score import calculate_wellness_score


# --------------------------------------------------
# 1. Load LifeLens data
# --------------------------------------------------

DATA_PATH = "data/daily_checkins.csv"

data = pd.read_csv(DATA_PATH)

if data.empty:
    print("No check-in data found.")
    exit()


# Use the latest check-in
latest = data.iloc[-1]


# --------------------------------------------------
# 2. Calculate the REAL LifeLens Wellness Score
# --------------------------------------------------

score, category = calculate_wellness_score(
    latest["SleepHours"],
    latest["Mood"],
    latest["Stress"],
    latest["Energy"],
    latest["ExerciseMinutes"],
    latest["ScreenTime"],
    latest["Productivity"],
)


# --------------------------------------------------
# 3. Load FLAN-T5-Small
# --------------------------------------------------

MODEL_NAME = "google/flan-t5-small"

print("\nLoading FLAN-T5-Small...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully.")


# --------------------------------------------------
# 4. Build LifeLens report prompt
# --------------------------------------------------

prompt = f"""
You are the wellness analyst for a personal wellness application called LifeLens.

Generate exactly 4 concise wellness report points based on the user's actual data.

Do not invent information.
Do not give medical advice.
Focus on the strongest patterns in the data.
Mention both positive aspects and areas that need improvement when appropriate.
The points should be different depending on the user's values.

Wellness Score: {score}/100
Wellness Category: {category}

Sleep: {latest["SleepHours"]} hours
Mood: {latest["Mood"]}/10
Energy: {latest["Energy"]}/10
Stress: {latest["Stress"]}/10
Exercise: {latest["ExerciseMinutes"]} minutes
Screen Time: {latest["ScreenTime"]} hours
Productivity: {latest["Productivity"]}/10

Write exactly 4 short numbered points.
"""


# --------------------------------------------------
# 5. Generate report
# --------------------------------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        num_beams=4,
        early_stopping=True
    )


report = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)


# --------------------------------------------------
# 6. Display result
# --------------------------------------------------

print("\n" + "=" * 60)
print("LIFELENS NLP WELLNESS REPORT")
print("=" * 60)

print(f"\nWellness Score: {score}/100")
print(f"Category: {category}")

print("\nGenerated Report:")
print(report)

print("\n" + "=" * 60)