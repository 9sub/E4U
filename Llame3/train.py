
import os
import torch
import argparse
from datasets import load_dataset

from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

import huggingface_hub

huggingface_hub.login('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')



def main(args):
    
    base_model = 'beomi/Llama-3-Open-Ko-8b'
    train_dental_dataset = "/Users/igyuseob/Desktop/AI/ET5/aihub_train_code/data_dir/train.jsonl"
    
    attn_implementation = "eager"
    torch_dtype = torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=False,
    )

    dataset = load_dataset('json', data_files=train_dental_dataset)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        #quantization_config=quant_config,
        device_map="auto"
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama3 training on custom dataset")

    parser.add_argument(
        "--train_file",
        type=str,
        default='/Users/igyuseob/Desktop/AI/ET5/aihub_train_code/data_dir/train.jsonl',
        help="Path to the existing train directory",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default='/Users/igyuseob/Desktop/AI/ET5/aihub_train_code/data_dir/vak.jsonl',
        help="Path to the existing validation directory",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default='/Users/igyuseob/Desktop/AI/ET5/aihub_train_code/data_dir/test.jsonl',
        help="Path to the test directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/Users/igyuseob/Desktop/AI/ET5/output',
        help="Path to the output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )

    args = parser.parse_args()
    main(args)