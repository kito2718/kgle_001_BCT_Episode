# Sprint 000: 基盤構築フェーズ (Foundation Setup)

- **スプリント期間**: 2026年11月25日 〜 (マイペース運用)
- **スプリントゴール**: ゼロトラスト高速検証器、ポートフォリオ管理基盤、VSCodeワンクリック開発環境を整備し、初期最良ベースライン(128k)を提出してLBと同期する。
- **役割**:
  - **PO (Product Owner)**: ユーザー様 (要件決定、優先順位判断、成果物レビュー)
  - **SM (Scrum Master)**: AI (プロセス管理、ブロッカー排除、振り返りファシリテーション)
  - **Developers**: ユーザー様 ＋ AI (実装、検証、探索)

---

## 🎯 完成の定義 (Definition of Done: DoD)

1. [x] **大会仕様・ルールの完全ドキュメント化**: `000_intro/` 配下に全タブの日本語解説が出力されていること。
2. [x] **ゼロトラスト高速検証器 (`verify_all.py`) の実装**:
   - 48レジスタ制約、未初期化レジスタ参照、負数/ゼロ減算、非整除除算、値上限超過($10^{18}$)、全ターゲット網羅を100%判定できること。
   - 1,000問の検証を0.13秒で完了。
3. [x] **Current Best Portfolio 管理機能**:
   - 1,000問の個別ベストスコアと命令列を管理し、1問でも改善があれば自動更新されること (`workspace/portfolio.py`)。
4. [x] **公開SOTA (128k) の統合と初期提出**:
   - 初期ポートフォリオを作成(128,219ステップ達成)、Kaggleへ提出してLBスコアと同期完了(128,219 / 341位)。
5. [x] **開発環境の整備**:
   - `workspace/` 直下にエントリポイント(`main.py`, `verify_all.py`, `portfolio.py`)を配置。
   - `workspace/output/` 配下にデータファイル(`puzzles.csv`, `practice_puzzles.csv`, `sample_submission.csv`)を集約。
   - `workspace/notebooks/` 配下に公開研究ノートブックを集約。
   - VSCode `tasks.json` によるワンクリック実行環境を配備。

---

## 📝 タスク一覧 & 進捗状況

| タスクID | 内容 | 担当 | 状態 | 成果物 / 備考 |
|---|---|:---:|:---:|---|
| **S000-0** | リポジトリ構造・開発プロセス初期化 | AI | Done | `README.md`, `PROJECT_DASHBOARD.md` |
| **S000-1** | 大会規約・仕様・全タブ日本語翻訳 | AI | Done | `000_intro/*.md` (全7ファイル) |
| **S000-2** | ゼロトラスト高速検証器の実装 | AI | Done | `workspace/verify_all.py`, `workspace/src/validator.py` |
| **S000-3** | 公開SOTAマージ & ポートフォリオ構築 | AI+PO | Done | `workspace/portfolio.py`, `workspace/output/best_portfolio.json` (128,219 steps) |
| **S000-4** | 初回Kaggle提出 & スコア整合確認 | PO+AI | Done | `workspace/output/submission.csv` (LB: 128,219 / 341位) |
| **S000-5** | VSCodeワンクリック実行タスク環境構築 | AI | Done | `workspace/.vscode/tasks.json` |

---

## 🔄 作業ログ (Work Log)

- **2026-11-25**:
  - Kaggle公式CLIおよびWebリソースから、Overview, Data, Code, Models, Discussion, Rulesの全コンテンツを取得。
  - `000_intro/` ディレクトリを作成し、全6タブの日本語翻訳および詳細解説ドキュメントを配置。
  - プロジェクト全体の「Scrumban × Experiment-Driven」運用体制および `PROJECT_DASHBOARD.md` を整備。
  - ユーザーからのフィードバックに基づき、フォルダ構成を最適化:
    - `data/` 配下の公式CSVを `workspace/output/` へ集約。
    - `notebooks/` (公開分析・参考ノートブック群) を `workspace/notebooks/` へ配置。
    - 実行エントリポイント(`main.py`, `verify_all.py`, `portfolio.py`)を `workspace/` 直下へ配置。
  - 公開最良フロンティア(128k SOTA)のプログラムを取り込み、ゼロトラスト検証器で完全検証(1000/1000パス)。
  - `best_portfolio.json` および提出用 `output/submission.csv` (128,219ステップ)を生成完了。
