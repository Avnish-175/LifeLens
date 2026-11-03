import os
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "nlp",
    "lifelens_nlp_model_v5_gpu"
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "lifelens_final_nlp_dataset_v5.csv"
)


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 80)
print("LIFELENS NLP - 20 EXAMPLE EVALUATION")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
model.eval()

print(f"Device: {device}")


# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv(DATA_FILE)

print(f"Dataset rows: {len(data)}")


# =========================================================
# TASKS
# =========================================================

tasks = [
    "daily_analysis",
    "weekly_analysis",
    "forecast_explanation",
    "daily_note_analysis"
]

N_EXAMPLES = 5


# =========================================================
# GENERATION
# =========================================================

def generate_prediction(model_input):

    inputs = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()


# =========================================================
# EVALUATION
# =========================================================

results = {}

for task_name in tasks:

    task_rows = data[
        data["task"].astype(str).str.strip() == task_name
    ]

    if len(task_rows) < N_EXAMPLES:
        raise ValueError(
            f"{task_name} has only {len(task_rows)} examples."
        )

    # Select 5 representative examples spread through dataset
    examples = task_rows.iloc[
        [int(i) for i in
         __import__("numpy").linspace(
             0,
             len(task_rows) - 1,
             N_EXAMPLES
         )]
    ]

    results[task_name] = []

    print("\n")
    print("#" * 80)
    print(f"TASK: {task_name.upper()}")
    print("#" * 80)

    for number, (_, row) in enumerate(
        examples.iterrows(),
        start=1
    ):

        input_text = str(row["input_text"]).strip()
        expected = str(row["target_text"]).strip()

        # EXACT TRAINING FORMAT
        model_input = (
            f"Task: {task_name}\n\n"
            f"{input_text}"
        )

        output = generate_prediction(model_input)

        results[task_name].append({
            "expected": expected,
            "output": output
        })

        print("\n" + "-" * 80)
        print(f"EXAMPLE {number}/5")
        print("-" * 80)

        print("\nINPUT:")
        print(model_input)

        print("\nEXPECTED:")
        print(expected)

        print("\nMODEL OUTPUT:")
        print(output)


# =========================================================
# SUMMARY
# =========================================================

print("\n\n")
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print("\nTotal examples tested: 20")
print("daily_analysis:        5")
print("weekly_analysis:       5")
print("forecast_explanation:  5")
print("daily_note_analysis:   5")

print("\nREAD-ONLY:")
print("- No training performed")
print("- No model files modified")
print("- No dataset files modified")
print("- No project files modified")
print("=" * 80)