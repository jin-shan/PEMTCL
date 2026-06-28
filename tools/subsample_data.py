#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 / 


1
    ToxiCN  FG-COLD  10%, 25%, 50%, 75% 
2 ToxiCN + FG-COLD train/dev/test 
   ToxiCN + FG-COLDRoBERTa


    # 
    python tools/subsample_data.py
    python tools/subsample_data.py --dataset toxicn --ratios 0.1 0.25 0.5
    python tools/subsample_data.py --seed 42 --output_suffix pct

    #  ToxiCN + FG-COLD  data/toxicn_cold_merged
    python tools/subsample_data.py --mode merge \\
        --merge_datasets toxicn fg-cold \\
        --merge_output_name toxicn_cold_merged \\
        --seed 41
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], file_path: Path):
    """JSON"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f" : {file_path} ({len(data)} )")


def subsample_dataset(
    dataset_name: str,
    data_root: Path,
    ratios: List[float],
    seed: int = 42,
    output_suffix: str = 'pct'
):
    """
    

    Args:
        dataset_name:  (toxicn/fg-cold)
        data_root: 
        ratios:  ( [0.1, 0.25, 0.5, 0.75])
        seed: 
        output_suffix: 
    """
    # 
    random.seed(seed)

    # 
    dataset_dir = data_root / dataset_name
    train_file = dataset_dir / 'train.json'
    test_file = dataset_dir / 'test.json'

    if not train_file.exists():
        print(f" : {train_file}  {dataset_name}")
        return

    # 
    full_train_data = load_json(train_file)
    print(f"\n{'='*60}")
    print(f": {dataset_name.upper()}")
    print(f": {len(full_train_data)} ")
    print(f"{'='*60}")

    # ()
    test_data = None
    if test_file.exists():
        test_data = load_json(test_file)
        print(f": {len(test_data)} ")

    # 
    for ratio in ratios:
        print(f"\n--- : {int(ratio*100)}% ---")

        # 
        sample_size = int(len(full_train_data) * ratio)

        # 
        sampled_train = random.sample(full_train_data, sample_size)

        #  lowres/Xpct/ 
        output_dir = data_root / f"{dataset_name}_lowres" / f"{int(ratio*100)}{output_suffix}"

        # 
        train_output = output_dir / 'train.json'
        save_json(sampled_train, train_output)

        # ()
        if test_data is not None:
            test_output = output_dir / 'test.json'
            save_json(test_data, test_output)

        # 
        print(f"  : {len(sampled_train)}  ({ratio*100:.1f}%)")
        if test_data:
            print(f"  (): {len(test_data)} ")


def merge_datasets(
    dataset_names: List[str],
    data_root: Path,
    output_name: str = 'toxicn_cold_merged',
    seed: int = 42,
    add_dataset_field: bool = True
):
    """
     train/dev/test 

    
    - data/toxicn, data/fg-fg-cold 
    -  data/<output_name>/  train/dev/test.json
    -  `dataset` 
    -  split  seed 
    """
    import random

    random.seed(seed)

    output_dir = data_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"")
    print(f": {', '.join(dataset_names)}")
    print(f": {output_dir}")
    print(f": {seed}")
    print(f"{'='*60}")

    for split in ['train', 'dev', 'test']:
        merged: List[Dict[str, Any]] = []
        total_before = 0

        for ds in dataset_names:
            src_file = data_root / ds / f'{split}.json'
            if not src_file.exists():
                print(f" : {src_file}  {ds}.{split}")
                continue

            records = load_json(src_file)
            total_before += len(records)

            if add_dataset_field:
                # 
                for r in records:
                    r_copy = dict(r)
                    #  dataset 
                    r_copy.setdefault('dataset', ds)
                    merged.append(r_copy)
            else:
                merged.extend(records)

        if not merged:
            print(f" :  {dataset_names}  {split}.json")
            continue

        # 
        random.shuffle(merged)

        out_file = output_dir / f'{split}.json'
        save_json(merged, out_file)

        print(f"   {split}: {len(merged)}  (: {total_before})")
        if add_dataset_field:
            #  dataset 
            from collections import Counter

            counter = Counter(r.get('dataset', 'unknown') for r in merged)
            print(f"  : {dict(counter)}")


def main():
    parser = argparse.ArgumentParser(
        description='/',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
:
  # 
  python tools/subsample_data.py

  #  ToxiCN
  python tools/subsample_data.py --dataset toxicn --ratios 0.1 0.2 0.5

  # 
  python tools/subsample_data.py --seed 123
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['subsample', 'merge', 'both'],
        default='both',
        help=': subsample=, merge=, both= (: both)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['toxicn', 'fg-cold', 'all'],
        default='all',
        help=' (: all)'
    )

    parser.add_argument(
        '--ratios',
        type=float,
        nargs='+',
        default=[0.1, 0.25, 0.5, 0.75],
        help=' (: 0.1 0.25 0.5 0.75)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help=' (: 42)'
    )

    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help=' (: data)'
    )

    parser.add_argument(
        '--output_suffix',
        type=str,
        default='pct',
        help=' (: pct)'
    )

    parser.add_argument(
        '--merge_datasets',
        type=str,
        nargs='+',
        default=['toxicn', 'fg-cold'],
        help=' merge  (: toxicn fg-cold)'
    )

    parser.add_argument(
        '--merge_output_name',
        type=str,
        default='toxicn_cold_merged',
        help=' data_root (: toxicn_cold_merged)'
    )

    parser.add_argument(
        '--merge_no_dataset_field',
        action='store_true',
        help=' dataset '
    )

    args = parser.parse_args()

    # 
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f" :  {data_root} ")
        return

    # 1mode=subsample  both 
    if args.mode in ('subsample', 'both'):
        # 
        if args.dataset == 'all':
            datasets = ['toxicn', 'fg-cold']
        else:
            datasets = [args.dataset]

        print(f"\n{'#'*60}")
        print(f"# ")
        print(f"#")
        print(f"# : {', '.join(datasets)}")
        print(f"# : {[f'{r*100}%' for r in args.ratios]}")
        print(f"# : {args.seed}")
        print(f"{'#'*60}")

        # 
        for dataset in datasets:
            subsample_dataset(
                dataset_name=dataset,
                data_root=data_root,
                ratios=args.ratios,
                seed=args.seed,
                output_suffix=args.output_suffix
            )

        print(f"\n{'='*60}")
        print(" !")
        print(f"{'='*60}\n")

    # 2mode=merge  both 
    if args.mode in ('merge', 'both'):
        #  data/toxicn_cold_merged
        merge_datasets(
            dataset_names=args.merge_datasets,
            data_root=data_root,
            output_name=args.merge_output_name,
            seed=args.seed,
            add_dataset_field=not args.merge_no_dataset_field
        )
        print(f"\n{'='*60}")
        print(" !")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

