# Purpose: Split the dataset into train, validation and test sets by disease-intention
import random
import shutil

from tqdm import tqdm
from pathlib import Path


def main(args):
    random.seed(args.seed)

    json_files = Path(args.data_dir).glob('**/*.json')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get parent directory path of json files
    json_parents = list(set([json_file.parent for json_file in json_files]))

    for json_dir in tqdm(json_parents):
        intention = json_dir.name
        disease = json_dir.parent.name
        category = json_dir.parent.parent.name
        train_dir = output_dir / 'train' / category / disease / intention
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir = output_dir / 'val' / category / disease / intention
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir = output_dir / 'test' / category / disease / intention
        test_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(json_dir.glob('*.json'))
        random.shuffle(json_files)
        num_files = len(json_files)
        num_train = int(num_files * args.train_ratio)
        num_val = int(num_files * args.val_ratio)
        num_test = num_files - num_train - num_val

        for i, json_file in enumerate(json_files):
            if i < num_train:
                shutil.copy(json_file, train_dir)
            elif i < num_train + num_val:
                shutil.copy(json_file, val_dir)
            else:
                shutil.copy(json_file, test_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Split the dataset into train, validation and test sets',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Path to the json data directory',
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training set',
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Ratio of validation set',
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Ratio of test set',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the output directory',
    )
    args = parser.parse_args()

    main(args)

