import argparse
from pathlib import Path
import json
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


def preprocess_function(examples, tokenizer):
    inputs = []
    labels = []

    for example in examples:
        question = example['question']
        answer  = example['answer']
        inputs.append(f"질문: {question}")
        labels.append(answer)

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    label_inputs = tokenizer(labels, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

    model_inputs["labels"] = label_inputs["input_ids"]
    return model_inputs

def load_data_from_dir(dir):
    all_examples = []
    for json_file in Path(dir).rglob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_examples.append({
                "question": data.get("질문", ""),
                "answer": data.get("답변", "")
            })
    return all_examples


def main(args):
    train_dir = args.train_dir
    val_dir = args.val_dir
    test_dir = args.test_dir
    output_dir = args.output_dir

    random.seed(args.seed)

    # 토크나이저 및 모델 불러오기
    tokenizer = T5Tokenizer.from_pretrained("ETRI/et5-base")
    model = T5ForConditionalGeneration.from_pretrained("ETRI/et5-base")

    # 데이터를 로드
    train_examples = load_data_from_dir(train_dir)
    val_examples = load_data_from_dir(val_dir)
    test_examples = load_data_from_dir(test_dir)

    # 데이터를 전처리
    train_dataset = preprocess_function(train_examples, tokenizer)
    val_dataset = preprocess_function(val_examples, tokenizer)
    test_dataset = preprocess_function(test_examples, tokenizer)

    # 학습 파라미터 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_steps=1000,
        save_total_limit=3,
        logging_dir='./logs',  # 로그를 기록할 경로
        logging_steps=500,
    )

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 검증 데이터셋 추가
        tokenizer=tokenizer,
    )

    # 모델 학습
    trainer.train()

    # 테스트 평가
    trainer.evaluate(eval_dataset=test_dataset)  # 테스트 데이터로 평가

    # 모델 저장
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ET5 training on custom dataset")

    parser.add_argument(
        "--train_dir",
        type=str,
        default='/Users/igyuseob/Desktop/capstone/data/120.초거대AI 사전학습용 헬스케어 질의응답 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터',
        help="Path to the existing train directory",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default='/Users/igyuseob/Desktop/capstone/data/120.초거대AI 사전학습용 헬스케어 질의응답 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터',
        help="Path to the existing validation directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default='/Users/igyuseob/Desktop/capstone/data/120.초거대AI 사전학습용 헬스케어 질의응답 데이터/3.개방데이터/1.데이터/test',
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