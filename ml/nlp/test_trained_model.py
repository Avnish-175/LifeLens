import os
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "nlp",
    "lifelens_nlp_model"
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "nlp_training_data.csv"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading trained LifeLens NLP model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
model.eval()

print(f"Device: {device}")


# ---------------------------------------------------------
# Load a real LifeLens example
# ---------------------------------------------------------

data = pd.read_csv(DATA_FILE)

example = data.sample(
    n=1,
    random_state=123
).iloc[0]

prompt = example["input_text"]

expected = example["target_text"]


# ---------------------------------------------------------
# Generate report
# ---------------------------------------------------------

inputs = tokenizer(
    prompt,
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

generated = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LIFELENS NLP MODEL TEST")
print("=" * 60)

print("\nINPUT:")
print(prompt)

print("\nMODEL GENERATED REPORT:")
print(generated)

print("\nEXPECTED TRAINING REPORT:")
print(expected)

print("\n" + "=" * 60)