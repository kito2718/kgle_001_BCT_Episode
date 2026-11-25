"""
Santa 2026 - The Elf Calculator: メインエントリポイント (CLI)

使用方法:
  python main.py verify [path]        : 提出ファイルまたはポートフォリオの採点・検証
  python main.py merge <path>         : 新しい解をベストポートフォリオにマージ・更新
  python main.py export <output_path> : 現在のベストポートフォリオから提出CSVを出力
  python main.py summary              : 現在のベストポートフォリオの状況を表示
"""

import sys
import os
import argparse
from verify_all import verify_submission
from portfolio import PortfolioManager


def main():
    parser = argparse.ArgumentParser(description="Santa 2026 Elf Calculator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify a submission CSV or portfolio JSON")
    verify_parser.add_argument("path", nargs="?", default="output/submission_sota_128221.csv", help="Path to file to verify")

    # merge
    merge_parser = subparsers.add_parser("merge", help="Merge candidates into current best portfolio")
    merge_parser.add_argument("path", help="Path to submission CSV or JSON")

    # export
    export_parser = subparsers.add_parser("export", help="Export portfolio to submission CSV")
    export_parser.add_argument("output", nargs="?", default="output/submission.csv", help="Output CSV path")

    # summary
    subparsers.add_parser("summary", help="Show summary of current best portfolio")

    args = parser.parse_args()

    if args.command == "verify":
        verify_submission(args.path)
    elif args.command == "merge":
        mgr = PortfolioManager()
        mgr.update_from_submission(args.path)
    elif args.command == "export":
        mgr = PortfolioManager()
        mgr.export_submission(args.output)
    elif args.command == "summary":
        mgr = PortfolioManager()
        mgr.print_summary()
    else:
        # デフォルトはサマリー表示または検証
        mgr = PortfolioManager()
        mgr.print_summary()


if __name__ == "__main__":
    main()
