import os
import math
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

import mlflow
import mlflow.pytorch
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "shortjokes.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"
TOKENIZER_OUTPUT_DIR = OUTPUT_DIR / "tokenizer"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"file://{BASE_DIR.parent / 'mlruns'}"
)

MODEL_NAME = os.getenv("BASE_MODEL_NAME", "gpt2")
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "text")
EPOCHS = int(os.getenv("TRAIN_EPOCHS", 1))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", 1))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 128))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 5e-5))
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "gpt2-joke-training")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = ["id", "text"]
    df = df[["text"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Dataset is too small. Add at least 10 rows.")

    # Smaller sample for fast local testing
    df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)

    return df


def tokenize_function(examples, tokenizer):
    # FIX 1: Added <|endoftext|> closing token — without it the model never
    # learns where a sequence ends, hurting generation quality.
    texts = [f"<|startoftext|> {t} <|endoftext|>" for t in examples[TEXT_COLUMN]]

    # FIX 2: Added labels mirror of input_ids so the Trainer can compute
    # the causal-LM loss. Without this the loss is None and training is a no-op.
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    df = load_data()

    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Force CPU (fix for Mac MPS issues)
    device = torch.device("cpu")
    model.to(device)

    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # FIX 3: Set dataset format to torch tensors so the DataLoader doesn't
    # receive raw Python lists, which causes a TypeError during collation.
    tokenized_train.set_format("torch")
    tokenized_eval.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "trainer"),
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        report_to=[],
        save_total_limit=1,
        no_cuda=True,  # force CPU on older transformers
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    with mlflow.start_run():
        mlflow.log_param("base_model", MODEL_NAME)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("eval_rows", len(eval_df))

        trainer.train()
        eval_metrics = trainer.evaluate()

        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # FIX 7: Added perplexity logging — it is the standard interpretability
        # metric for language models and was missing from MLflow tracking.
        eval_loss = eval_metrics.get("eval_loss")
        if eval_loss is not None:
            mlflow.log_metric("perplexity", math.exp(eval_loss))

        model.save_pretrained(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)

        mlflow.log_artifacts(str(MODEL_OUTPUT_DIR), artifact_path="model")
        mlflow.log_artifacts(str(TOKENIZER_OUTPUT_DIR), artifact_path="tokenizer")

        print(f"Training complete.")
        print(f"Model saved to: {MODEL_OUTPUT_DIR}")
        print(f"Tokenizer saved to: {TOKENIZER_OUTPUT_DIR}")
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
