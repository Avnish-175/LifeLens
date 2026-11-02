import os
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "nlp_training_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "nlp",
    "lifelens_nlp_model"
)

BASE_MODEL = "google/flan-t5-small"


# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------

print("=" * 60)
print("LOADING LIFELENS NLP TRAINING DATA")
print("=" * 60)

data = pd.read_csv(DATA_FILE)

data = data.dropna(
    subset=["input_text", "target_text"]
).reset_index(drop=True)

print(f"Training examples: {len(data)}")


# ---------------------------------------------------------
# Convert to HuggingFace Dataset
# ---------------------------------------------------------

dataset = Dataset.from_pandas(
    data[["input_text", "target_text"]],
    preserve_index=False
)


# ---------------------------------------------------------
# Load tokenizer and model
# ---------------------------------------------------------

print("\nLoading FLAN-T5-Small...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL
)


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize_function(batch):

    inputs = tokenizer(
        batch["input_text"],
        max_length=256,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=160,
        truncation=True,
        padding=False,
    )

    inputs["labels"] = labels["input_ids"]

    return inputs


print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)


# ---------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------

split_dataset = tokenized_dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training rows:   {len(train_dataset)}")
print(f"Validation rows: {len(eval_dataset)}")


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=MODEL_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    learning_rate=5e-5,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=50,

    load_best_model_at_end=True,

    report_to="none",

    fp16=torch.cuda.is_available(),
)


# ---------------------------------------------------------
# Data collator
# ---------------------------------------------------------

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)


# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,
)


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING LIFELENS NLP TRAINING")
print("=" * 60)

trainer.train()


# ---------------------------------------------------------
# Save final model
# ---------------------------------------------------------

print("\nSaving LifeLens NLP model...")

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Model saved to:")
print(MODEL_DIR)