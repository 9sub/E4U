from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import pandas as pd
from datasets import Dataset
import argparse
import torch
from accelerate import Accelerator
import datetime
from tqdm.auto import tqdm

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 

# Initialize Accelerator
accelerator = Accelerator()

# Argument parser for script parameters
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="/Users/igyuseob/Downloads/초거대 AI 헬스케어 질의응답 데이터/1.모델/2.AI학습모델파일/질의응답학습모델")
parser.add_argument("--domain", type=str, default="daily")
parser.add_argument("--checkpoint_path", type=str, default="./output")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)

args = parser.parse_args()
model_id = args.model_id
checkpoint_path = args.checkpoint_path
epochs = args.epochs
batch_size = args.batch_size

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

# Load the data
train_file = "/Users/igyuseob/Desktop/capstone/data/일상대화 한국어 멀티세션 데이터/1.model source code/chatbot_data/chatbot_data_train.tsv"
val_file = "/Users/igyuseob/Desktop/capstone/data/일상대화 한국어 멀티세션 데이터/1.model source code/chatbot_data/chatbot_data_val.tsv"
test_file = "/Users/igyuseob/Desktop/capstone/data/일상대화 한국어 멀티세션 데이터/1.model source code/chatbot_data/chatbot_data_test.tsv"

# Preprocess the data
def preprocess_function(data):
    model_inputs = tokenizer(
        data["user_input"], max_length=1024, padding="max_length", truncation=True
    )
    model_inputs["labels"] = tokenizer(
        data["system_output"], max_length=300, padding="max_length", truncation=True
    )["input_ids"]
    return model_inputs

# Load and preprocess datasets
train_df = pd.read_csv(train_file, delimiter='\t')
val_df = pd.read_csv(val_file, delimiter='\t')
test_df = pd.read_csv(test_file, delimiter='\t')

train_dataset = Dataset.from_pandas(train_df).map(preprocess_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(preprocess_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(preprocess_function, batched=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoint_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_dir=f"{checkpoint_path}/logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # or use any other metric
    save_total_limit=2,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize the Trainer with tqdm progress
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training with progress tracking
print(f"Start time: {datetime.datetime.now()}")
trainer.train()

# Evaluate on the test set and display tqdm progress
print("Evaluating on test data...")
test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
for key, value in test_results.items():
    print(f"{key}: {value}")

print(f"End time: {datetime.datetime.now()}")

# Save the final model checkpoint
trainer.save_model(f"{checkpoint_path}/final-checkpoint")