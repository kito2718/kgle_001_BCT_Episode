# 04. Models (最適化モデル & ソルバー)

Kaggle公式ページ: [Santa 2026 - The Elf Calculator (Models)](https://www.kaggle.com/competitions/santa-2026/models)

---

## 1. 本コンペにおける「モデル(Model)」の位置づけ

本コンペティション「Santa 2026 - The Elf Calculator」は、従来の画像認識や自然言語処理のような教師あり学習(Supervised Learning)ではなく、**離散数学・組み合わせ最適化・コードゴルフ(Code-Golf)** の領域に属するコンペティションです。

そのため、Kaggleの「Models」タブにアップロードされるPyTorchやTensorFlow等の重みファイルではなく、**数理最適化モデル(Mathematical Programming Models)、制約充足ソルバー(Constraint Programming)、グラフ探索モデル、および探索用LLMエージェント**が本質的な「Model」としての役割を果たします。

---

## 2. 活躍する主要な最適化モデル・ソルバー

### (1) 制約プログラミング / SATソルバー (CP-SAT / MIP)
- **代表的ツール**: Google OR-Tools CP-SAT, Gurobi, Z3 SMT Solver, SCIP
- **適用先**:
  - **複数プログラムのリコンビネーション**: 異なる探索によって見つかった有効な中間計算値(候補ノード)をプールし、「すべてのターゲットをカバーしつつ、総ノード数を最小化する」集合被覆問題(Set Covering / Minimum Weight Steiner Tree)として定式化して解く。
  - **局所サブグラフの最適合成**: 3〜4手以内の小規模な中間グラフについて、CP-SATで厳密な最短手を保証する。

### (2) グラフ探索モデル (Graph Search & Beam Search)
- **前方・後退ビームサーチ (Forward / Backward Beam Search)**:
  - 状態空間: レジスタプール内の保持値
  - 遷移: 4則演算による新たな値の生成
  - 評価関数(ヒューリスティック): 目標値とのビット距離、残余ターゲット数、因数分解可能性
- **A* 探索 / 双方向探索 (Bidirectional Search)**:
  - 初期値集合から前向きに展開するフロンティアと、ターゲット値から逆算展開するフロンティアの衝突(Meet-in-the-middle)を検出。

### (3) メタヒューリスティクス・局所探索モデル (LNS / Simulated Annealing)
- **大規模近傍探索 (Large Neighborhood Search: LNS)**:
  - 破壊(Destruction): 既存の解から一部のノードを意図的に削除。
  - 修復(Repair): 残されたノード群から貪欲法または全探索で最短パスを再構築。
- **焼きなまし法 (Simulated Annealing: SA)**:
  - レジスタスロットの順序付け(トポロジカルソート順)やヘルパーノードの選択において、一時的な改悪や同等長(Plateau)を確率的に受容して大域最適解を探索。

### (4) LLMコード探索・自動最適化エージェント (Agentic Code Synthesis)
- 近年急速に注目されているアプローチ。
- Claude 3.5 Sonnet / Gemini Pro などのLLMに対し、ターゲット値の素因数分解、差分、共通ハブの数学的分析を行わせ、カスタムの生成アルゴリズムや新しい探索ヒューリスティクスを自動設計・コード生成させるアプローチ。
- 公開ノートブックでも「Claude Coding」「LLM-driven Discovery」として大きな成果を上げています。

---

## 3. レジスタ制約モデル (Pebble Game & Register Allocation)

DAG(有向非巡回グラフ)が求まった後、それを「48レジスタ以内で実行可能な命令列」にマッピングするコンパイラ最適化モデルです。

1. **レジスタ割り当てグラフ彩色 (Graph Coloring / Linear Scan)**:
   - 各中間値の生存区間(Live Range)を分析し、同時にアクティブな値が最大48個を超えないようにレジスタ番号(0〜47)を割り当て。
2. **ペブルゲーム(Pebble Game)によるスケジューリング**:
   - 48個の「小石(Pebble)」を使ってグラフを走査する理論。中間値を破棄するタイミングと再計算のトレードオフをモデル化。

---

## 関連リンク
- **[03_code.md](./03_code.md)**: 公開コード動向と解法アプローチ
- **[05_discussion.md](./05_discussion.md)**: コミュニティ議論と知見まとめ
- **[PROJECT_DASHBOARD.md](../PROJECT_DASHBOARD.md)**: プロジェクトダッシュボード
