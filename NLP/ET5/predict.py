# Generate answer from input text
# Author: MEDIAZEN AIMZ R&D Group NLP Team

import re
import os
import sys
import random
import argparse
import torch

from pathlib import Path
from typing import List, Dict, Any
sys.path.append(
    str(Path(__file__).parent.parent)
)
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model(model_path):
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer(
        vocab_file=f"{model_path}/spiece.model",
        config=f"{model_path}/config.json",
    )
    #tokenizer = T5Tokenizer(vocab_file='/Users/igyuseob/Downloads/1_et5_download_mask_iii_base/spiece.model', config='/Users/igyuseob/Downloads/1_et5_download_mask_iii_base/config.json')

    return model, tokenizer

def main(args):
    model, tokenizer = load_model(args.model_path)
    model.eval()

    input_text = input("input text: ")
    input_ids = tokenizer(
        input_text,
        return_tensors='pt',
    ).input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            top_p=1,
        )

    decoded_output = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return decoded_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="model path",
    )
    args = parser.parse_args()

    main(args)
