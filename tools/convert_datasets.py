#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert external datasets into MTL-compatible JSON."""

import json
from ast import literal_eval
from pathlib import Path
from typing import List

import pandas as pd


def parse_list(value, expected_len: int) -> List[int]:
    if isinstance(value, list):
        parsed = value
    else:
        parsed = literal_eval(str(value))
    parsed = [int(x) for x in parsed]
    if len(parsed) != expected_len:
        raise ValueError(f"List length {len(parsed)} != expected {expected_len}: {parsed}")
    return parsed


def write_split(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def convert_cold(data_dir: Path, output_dir: Path):
    def load_split(filename: Path):
        df = pd.read_csv(filename)
        records = []
        for _, row in df.iterrows():
            content = str(row['content']).strip()
            record = {
                'platform': row.get('platform', 'fg-cold'),
                'topic': row.get('topic', ''),
                'content': content,
                'toxic': int(row['toxic']),
                'toxic_one_hot': parse_list(row['toxic_one_hot'], 2),
                'toxic_type': int(row['toxic_type']),
                'toxic_type_one_hot': parse_list(row['toxic_type_one_hot'], 2),
                'expression': int(row['expression']),
                'expression_one_hot': parse_list(row['expression_one_hot'], 3),
                'target': parse_list(row['target'], 5),
                'length': len(content)
            }
            records.append(record)
        return records

    write_split(load_split(data_dir / 'train.csv'), output_dir / 'train.json')
    write_split(load_split(data_dir / 'dev.csv'), output_dir / 'dev.json')
    write_split(load_split(data_dir / 'test.csv'), output_dir / 'test.json')


def main():
    base = Path(__file__).resolve().parent.parent
    cold_src = base / 'data' / 'fg-cold'
    convert_cold(cold_src, cold_src)


if __name__ == '__main__':
    main()

