# システムアーキテクチャ設計書 (System Architecture)

本ドキュメントは、「Santa 2026 - The Elf Calculator」において、世界1位を獲得するためのソルバーアーキテクチャおよびデータパイプラインの全体設計を定義します。

---

## 1. 全体アーキテクチャ概要

```mermaid
flowchart TD
    subgraph DataLayer["データ・ポートフォリオ層"]
        PuzzlesCSV["data/puzzles.csv<br/>(1,000問の仕様)"]
        PortfolioDB["data/best_portfolio.json<br/>(1,000問の現行最短解)"]
        SubCSV["output/submission.csv<br/>(Kaggle提出用CSV)"]
    end

    subgraph CoreEngine["コア実行・検証層 (Zero-Trust)"]
        Validator["validator.py<br/>(48レジスタ・厳格ハードウェアシミュレータ)"]
        Referee["verify_all.py<br/>(1,000問一括採点・ルール違反即時検知)"]
    end

    subgraph RepresentationLayer["グラフ抽象化層"]
        Prog2DAG["program_to_dag.py<br/>(命令列 ➔ Value Graph)"]
        DAG2Prog["dag_to_program.py<br/>(Value Graph ➔ 48レジスタ命令列)"]
        RegAlloc["register_allocator.py<br/>(Pebble Game / 線形走査法)"]
    end

    subgraph OptimizationLayer["多角探索エンジン群 (Lineages)"]
        LNSEngine["engine_lns.py<br/>(Value-Graph LNS / Cone Rebuild)"]
        BeamEngine["engine_beam.py<br/>(Backward Beam Search)"]
        HubEngine["engine_hub.py<br/>(共通因数・GCD・基底探索)"]
        CPSATEngine["engine_recombine.py<br/>(CP-SAT 複数解リコンビネーション)"]
    end

    PuzzlesCSV --> Validator
    PortfolioDB --> Referee
    Validator --> Referee
    
    PortfolioDB --> Prog2DAG
    Prog2DAG --> OptimizationLayer
    OptimizationLayer --> DAG2Prog
    DAG2Prog --> RegAlloc
    RegAlloc --> Validator
    Validator -->|スコア改善確認時のみ| PortfolioDB
    PortfolioDB --> SubCSV
```

---

## 2. モジュール構成と責務

1. **`src/validator.py` (計算機エミュレータ)**:
   - 48レジスタの仮想マシンを忠実に再現。
   - レジスタ範囲(0〜47)、未初期化参照、正の整数制約、除算の整除性($\pmod == 0$)、値上限($10^{18}$)、全ターゲット到達を検証。
2. **`src/portfolio.py` (ポートフォリオマネージャ)**:
   - 1,000パズルのベストプログラムを保持。
   - 新しい解が生成された際、検証器をパスし、かつ既存のステップ数より短い場合のみ該当パズルを置き換える。
   - 提出用CSV(`submission.csv`)の自動生成とJSONエクスポート。
3. **`src/dag.py` (Value Graph抽象化)**:
   - 命令列からレジスタ割り当ての依存を取り除き、純粋な値の依存関係(DAG)として表現。
4. **`src/reg_alloc.py` (レジスタアロケータ)**:
   - DAGのトポロジカルソート順を最適化し、最大同時生存値数を48個以下に抑えて具体的なレジスタ番号(0〜47)を割り当てる。

---

## 関連リンク
- **[02_optimization_strategy.md](./02_optimization_strategy.md)**: 最適化アルゴリズム戦略
- **[PROJECT_DASHBOARD.md](../PROJECT_DASHBOARD.md)**: プロジェクトダッシュボード
