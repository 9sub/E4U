import random
import shutil

from pathlib import Path

def main(args):
    random.seed(args.seed)

    train_dir = Path(args.data_dir) / 'training/02.라벨링데이터'
    output_dir = Path(args.data_dir) / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(train_dir.glob('**/*.json'))
    random.shuffle(json_files)

    num_files = len(json_files)
    num_test = int(num_files * args.test_ratio)

    test_dir = output_dir / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)

    for(i, json_file) in enumerate(json_files):
        if(i < num_test):
            relative_path = json_file.relative_to(train_dir)
            target_path = test_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(json_file), str(target_path))
        else:
            continue

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Split the existing dataset to create a test dataset'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Path to the existing train directory'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Ratio of the test dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the output directory'
    )
    args = parser.parse_args()
    main(args)