# Purpose : Make training data for the model

import json

from tqdm import tqdm
from pathlib import Path

from typing import Literal
from itertools import cycle
from concurrent.futures import ProcessPoolExecutor


def read_json(file_path: Path) -> dict:
    return json.load(file_path.open('r'))

def process_batch(file_batch):
    """배치 데이터를 읽어서 질문과 답변을 반환"""
    qa_data = []
    for file_path in file_batch:
        data = read_json(file_path)
        if file_path.name.startswith("HC-A"):
            qa_data.append({"desease": data["disease_name"]["kor"], "intention": data["intention"], "text": get_answer(data), "type": "answer"})
        else:
            qa_data.append({"desease": data["disease_name"]["kor"], "intention": data["intention"], "text": data["question"], "type": "question"})
    return qa_data

def batch_files(file_paths, batch_size=100):
    """파일 리스트를 배치 사이즈로 나누어서 반환"""
    for i in range(0, len(file_paths), batch_size):
        yield file_paths[i:i+batch_size]

def write_jsonl(question: str, answer: str, file_path: Path) -> None:
    """질문과 답변을 jsonl 파일에 쓰기"""
    with Path(file_path).open('a') as f:
        f.write(json.dumps({"text": question, "label": answer}, ensure_ascii=False) + '\n')

def get_answer(answer_data: dict) -> str:
    """답변 데이터에서 답변 텍스트를 추출"""
    answer = (
        answer_data["answer"]["intro"]
        + '\n'
        + answer_data["answer"]["body"]
        + '\n'
        + answer_data["answer"]["conclusion"]
    )
    return answer


def main(args):
    # train, val, test
    splits = {"train": "train", 
            "val": "val", 
            "test": "test"}

    # Read question and answer json files
    raw_data_dir = Path(args.raw_data_dir) 
    output_dir = Path(args.output_dir)

    # Make output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    used_question_files = []
    for split in splits.keys():
        fname = splits[split]
        output_file = output_dir / Path(f"{fname}.jsonl")

        # Read answer json files
        answers = list(raw_data_dir.glob(f"{split}/02.labeling_data/1.question/**/*.json"))
        questions = list(raw_data_dir.glob(f"{split}/02.labeling_data/2.answer/**/*.json"))
        print(split)
        # File list
        total = answers + questions

        # Creating maps for questions and answers
        qa_map = {}

        # Processing batch
        print(f"Processing", raw_data_dir, split)
        with ProcessPoolExecutor(max_workers=8) as executor:
            for file_batch in batch_files(total):
                batch_results = list(executor.map(process_batch, [file_batch]))
                print(batch_results)
                for batch_result in batch_results:
                    for item in batch_result:
                        print(item)
                        key = (item["desease"], item["intention"])
                        if key not in qa_map:
                            qa_map[key] = {"questions": [], "answers": []}
                        if item["type"] == "question":
                            qa_map[key]["questions"].append(item["text"])
                        else:
                            qa_map[key]["answers"].append(item["text"])

        # Pairing questions and answers
        for key, items in tqdm(qa_map.items()):
            qs = items["questions"]
            ans = items["answers"]

            ans_cycle = cycle(ans) if len(ans) < len(qs) else iter(ans)
            qs_cycle = cycle(qs) if len(qs) < len(ans) else iter(qs)

            for question, answer in zip(qs_cycle, ans_cycle):
                output_file = output_dir / Path(f"{fname}.jsonl")
                write_jsonl(question, answer, output_file)

    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)
