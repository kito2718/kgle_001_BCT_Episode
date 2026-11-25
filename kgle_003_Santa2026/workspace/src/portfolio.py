"""
Current Best Portfolio Manager
1,000問それぞれの最短プログラムを保持・更新・提出CSV生成するマネージャ
"""

import os
import sys
import json
import argparse
import pandas as pd
from validator import ElfCalculatorValidator


class PortfolioManager:
    def __init__(self, puzzles_path: str = "../data/puzzles.csv", portfolio_path: str = "../data/best_portfolio.json"):
        self.puzzles_path = puzzles_path
        self.portfolio_path = portfolio_path
        self.validator = ElfCalculatorValidator(puzzles_path)
        self.portfolio = {}
        self.load_portfolio()

    def load_portfolio(self):
        if os.path.exists(self.portfolio_path):
            with open(self.portfolio_path, 'r', encoding='utf-8') as f:
                self.portfolio = json.load(f)
            print(f"Loaded existing portfolio with {len(self.portfolio)} puzzles.")
        else:
            print("No existing portfolio found. Initializing empty.")
            self.portfolio = {}

    def save_portfolio(self):
        with open(self.portfolio_path, 'w', encoding='utf-8') as f:
            json.dump(self.portfolio, f, indent=None)
        print(f"Portfolio saved to {self.portfolio_path}")

    def update_from_submission(self, sub_path: str):
        """
        提出CSVまたはJSONからプログラムを読み込み、短縮されたものがあれば更新する
        """
        print(f"Evaluating candidates from: {sub_path}")
        if sub_path.endswith('.csv'):
            df = pd.read_csv(sub_path)
            candidates = []
            for _, row in df.iterrows():
                pid = str(row['id']).strip()
                prog = json.loads(row['program']) if isinstance(row['program'], str) else row['program']
                candidates.append((pid, prog))
        else:
            with open(sub_path, 'r', encoding='utf-8') as f:
                candidates = list(json.load(f).items())

        improved_count = 0
        total_saved_steps = 0

        for pid, prog in candidates:
            valid, msg, steps = self.validator.validate_puzzle(pid, prog)
            if not valid:
                print(f"Warning: Candidate for {pid} is invalid: {msg}. Skipped.")
                continue

            current_steps = len(self.portfolio[pid]) if pid in self.portfolio else float('inf')
            if steps < current_steps:
                saved = current_steps - steps if current_steps != float('inf') else 0
                self.portfolio[pid] = prog
                improved_count += 1
                total_saved_steps += saved

        print(f"Update summary: {improved_count} puzzles improved, saved {total_saved_steps} total steps!")
        self.save_portfolio()
        self.print_summary()

    def print_summary(self):
        if not self.portfolio:
            print("Portfolio is empty.")
            return

        total_steps = sum(len(prog) for prog in self.portfolio.values())
        print("=" * 50)
        print(f"CURRENT PORTFOLIO SUMMARY")
        print(f"Covered Puzzles: {len(self.portfolio)} / 1000")
        print(f"TOTAL SCORE (Steps): {total_steps}")
        print(f"Average Steps/Puzzle: {total_steps / len(self.portfolio):.2f}")
        print("=" * 50)

    def export_submission(self, output_csv_path: str):
        if len(self.portfolio) < 1000:
            raise ValueError(f"Portfolio is incomplete ({len(self.portfolio)}/1000). Cannot export valid submission.")

        rows = []
        for i in range(1, 1001):
            pid = f"scored_{i:04d}"
            if pid not in self.portfolio:
                raise KeyError(f"Missing puzzle {pid} in portfolio")
            prog_json = json.dumps(self.portfolio[pid], separators=(',', ':'))
            rows.append({'id': pid, 'program': prog_json})

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Exported clean submission to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Current Best Portfolio Manager")
    parser.add_argument("--merge", type=str, help="Merge new solution CSV/JSON into portfolio")
    parser.add_argument("--export-submission", type=str, help="Export portfolio to submission CSV")
    parser.add_argument("--summary", action="store_true", help="Print current portfolio summary")
    args = parser.parse_args()

    mgr = PortfolioManager()
    if args.merge:
        mgr.update_from_submission(args.merge)
    elif args.export_submission:
        mgr.export_submission(args.export_submission)
    else:
        mgr.print_summary()
