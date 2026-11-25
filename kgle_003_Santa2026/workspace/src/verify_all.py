"""
全1,000問の提出ファイルまたはポートフォリオの一括検証・採点スクリプト
"""

import sys
import os
import json
import time
import pandas as pd
from validator import ElfCalculatorValidator


def find_puzzles_path():
    candidates = [
        "../data/puzzles.csv",
        "data/puzzles.csv",
        "../../data/puzzles.csv",
        "puzzles.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find data/puzzles.csv")


def verify_submission(csv_or_json_path: str):
    puzzles_path = find_puzzles_path()
    print(f"Loading puzzles from: {puzzles_path}")
    validator = ElfCalculatorValidator(puzzles_path)

    print(f"Verifying: {csv_or_json_path}")
    start_time = time.time()

    if csv_or_json_path.endswith('.csv'):
        df = pd.read_csv(csv_or_json_path)
        items = []
        for _, row in df.iterrows():
            pid = str(row['id']).strip()
            prog = json.loads(row['program']) if isinstance(row['program'], str) else row['program']
            items.append((pid, prog))
    elif csv_or_json_path.endswith('.json'):
        with open(csv_or_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = list(data.items())
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")

    total_steps = 0
    passed_count = 0
    failed_puzzles = []

    for pid, prog in items:
        valid, msg, steps = validator.validate_puzzle(pid, prog)
        if valid:
            total_steps += steps
            passed_count += 1
        else:
            failed_puzzles.append((pid, msg))

    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"Verification Finished in {elapsed:.2f} seconds")
    print(f"Total Puzzles: {len(items)}")
    print(f"Passed: {passed_count} / {len(items)}")
    
    if failed_puzzles:
        print(f"FAILED: {len(failed_puzzles)} puzzles!")
        for pid, msg in failed_puzzles[:10]:
            print(f"  - [{pid}] {msg}")
        if len(failed_puzzles) > 10:
            print(f"  ... and {len(failed_puzzles) - 10} more.")
        print("=" * 60)
        sys.exit(1)
    else:
        print("ALL PUZZLES VALID! (Zero Errors)")
        print(f"TOTAL SCORE (Steps): {total_steps}")
        print(f"Average Steps/Puzzle: {total_steps / len(items):.2f}")
        print("=" * 60)
        return total_steps


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "../data/sample_submission.csv"
    verify_submission(target)
