import os
import sys
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "lifelens_final_nlp_dataset_v5.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "nlp",
    "lifelens_nlp_model_v5_gpu"
)

BASE_MODEL = "google/flan-t5-small"


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128

RANDOM_SEED = 42


# ============================================================
# GPU SETUP
# ============================================================

print("=" * 60)
print("LIFELENS NLP TRAINING - GPU VERSION")
print("=" * 60)

print("\nPyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    GPU_NAME = torch.cuda.get_device_name(0)

    GPU_MEMORY = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print("GPU:", GPU_NAME)
    print("GPU VRAM:", round(GPU_MEMORY, 2), "GB")

    # Allow TF32 where supported.
    # This does NOT disable CUDA.
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Training device: CUDA")

else:

    DEVICE = torch.device("cpu")

    print("GPU not available.")
    print("Training device: CPU")


# ============================================================
# DATASET VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("LOADING LIFELENS NLP DATASET V5")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    print("\nERROR: Dataset not found:")
    print(DATA_PATH)
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

print("\nDataset path:")
print(DATA_PATH)

print("\nTotal examples:", len(df))

print("\nColumns:")
print(list(df.columns))


# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "task",
    "input_text",
    "target_text",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing required columns:")
    print(missing_columns)

    sys.exit(1)


# ------------------------------------------------------------
# REMOVE EMPTY ROWS
# ------------------------------------------------------------

df = df.dropna(
    subset=[
        "task",
        "input_text",
        "target_text",
    ]
).copy()


# Convert to strings

df["task"] = df["task"].astype(str)
df["input_text"] = df["input_text"].astype(str)
df["target_text"] = df["target_text"].astype(str)


# ------------------------------------------------------------
# VALID TASKS
# ------------------------------------------------------------

valid_tasks = {
    "daily_analysis",
    "weekly_analysis",
    "forecast_explanation",
    "daily_note_analysis",
}

invalid_tasks = set(df["task"].unique()) - valid_tasks

if invalid_tasks:

    print("\nERROR: Invalid task names:")
    print(invalid_tasks)

    sys.exit(1)


# ------------------------------------------------------------
# EMPTY TEXT CHECK
# ------------------------------------------------------------

empty_inputs = (
    df["input_text"].str.strip().str.len() == 0
).sum()

empty_targets = (
    df["target_text"].str.strip().str.len() == 0
).sum()

if empty_inputs > 0:

    print(
        f"\nERROR: {empty_inputs} empty input_text rows found."
    )

    sys.exit(1)

if empty_targets > 0:

    print(
        f"\nERROR: {empty_targets} empty target_text rows found."
    )

    sys.exit(1)


# ------------------------------------------------------------
# DUPLICATE CHECK
# ------------------------------------------------------------

duplicates = df.duplicated(
    subset=[
        "task",
        "input_text",
        "target_text",
    ]
).sum()

print("\nDuplicate rows:", duplicates)

if duplicates > 0:

    print("Removing duplicate rows...")

    df = df.drop_duplicates(
        subset=[
            "task",
            "input_text",
            "target_text",
        ]
    ).reset_index(drop=True)


# ============================================================
# TASK DISTRIBUTION
# ============================================================

print("\nTask distribution:")

print(
    df["task"]
    .value_counts()
)


# ============================================================
# LOAD TOKENIZER + MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING FLAN-T5-SMALL")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL
)

print("\nModel loaded successfully.")


# ============================================================
# MOVE MODEL TO GPU
# ============================================================

model.to(DEVICE)

print("\nModel device:")

print(
    next(model.parameters()).device
)


# ============================================================
# BUILD TRAINING INPUT
# ============================================================

def build_training_input(row):

    return (
        f"Task: {row['task']}\n\n"
        f"{row['input_text']}"
    )


df["training_input"] = df.apply(
    build_training_input,
    axis=1
)


# ============================================================
# CONVERT TO HUGGINGFACE DATASET
# ============================================================

dataset = Dataset.from_pandas(
    df[
        [
            "training_input",
            "target_text",
        ]
    ],
    preserve_index=False
)


# ============================================================
# TOKENIZATION
# ============================================================

print("\n" + "=" * 60)
print("TOKENIZING DATASET")
print("=" * 60)


def tokenize_function(examples):

    inputs = tokenizer(
        examples["training_input"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    inputs["labels"] = labels["input_ids"]

    return inputs


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[
        "training_input",
        "target_text",
    ],
)


# ============================================================
# TOKENIZATION SANITY CHECK
# ============================================================

print("\nChecking tokenized dataset...")

sample = tokenized_dataset[0]

print(
    "Sample input token count:",
    len(sample["input_ids"])
)

print(
    "Sample label token count:",
    len(sample["labels"])
)

if len(sample["input_ids"]) == 0:

    print("\nERROR: Empty input tokens.")

    sys.exit(1)

if len(sample["labels"]) == 0:

    print("\nERROR: Empty label tokens.")

    sys.exit(1)

print("Tokenization check: PASSED")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split_dataset = tokenized_dataset.train_test_split(
    test_size=0.10,
    seed=RANDOM_SEED
)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


print("\nTraining rows:  ", len(train_dataset))
print("Validation rows:", len(eval_dataset))


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=MODEL_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    

    # IMPORTANT:
    # FP16 is disabled because the previous run produced:
    #
    # loss = 0
    # grad_norm = nan
    #
    # The GPU is STILL being used.
    fp16=False,

    # Do not use BF16 on this RTX 2050.
    bf16=False,

    logging_strategy="steps",

    logging_steps=50,

    eval_strategy="epoch",

    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=False,

    report_to="none",

    seed=RANDOM_SEED,

    dataloader_pin_memory=torch.cuda.is_available(),

    # Gradient clipping for additional stability.
    max_grad_norm=1.0,

)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING LIFELENS NLP TRAINING")
print("=" * 60)

print("\nConfiguration:")
print("Base model:", BASE_MODEL)
print("Epochs:", NUM_EPOCHS)
print("Train batch:", TRAIN_BATCH_SIZE)
print("Eval batch:", EVAL_BATCH_SIZE)
print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS
)
print("Learning rate:", LEARNING_RATE)
print("FP16:", False)
print("BF16:", False)
print("Device:", DEVICE)

print("\nEffective batch size:")

print(
    TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)


# ============================================================
# TRAIN
# ============================================================

try:

    train_result = trainer.train()

except RuntimeError as error:

    print("\n" + "=" * 60)
    print("TRAINING ERROR")
    print("=" * 60)

    print(error)

    if torch.cuda.is_available():

        print("\nCUDA memory information:")

        print(
            torch.cuda.memory_summary(
                device=0,
                abbreviated=True
            )
        )

    raise


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nTraining metrics:")

print(
    train_result.metrics
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

trainer.save_model(
    MODEL_DIR
)

tokenizer.save_pretrained(
    MODEL_DIR
)

print("\nModel saved to:")

print(MODEL_DIR)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

evaluation_results = trainer.evaluate()

print("\nEvaluation results:")

for key, value in evaluation_results.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    print("\n" + "=" * 60)
    print("GPU MEMORY SUMMARY")
    print("=" * 60)

    allocated = (
        torch.cuda.memory_allocated(0)
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved(0)
        / (1024 ** 3)
    )

    print(
        "GPU memory allocated:",
        round(allocated, 2),
        "GB"
    )

    print(
        "GPU memory reserved:",
        round(reserved, 2),
        "GB"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("LIFELENS NLP TRAINING COMPLETE")
print("=" * 60)

print("\nModel directory:")

print(MODEL_DIR)

print("\nNext step:")

print(
    "Run the NLP evaluation script against "
    "lifelens_nlp_model_v5_gpu."
)