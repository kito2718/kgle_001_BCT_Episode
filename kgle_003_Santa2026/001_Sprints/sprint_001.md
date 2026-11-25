# Sprint 001: 学術知見整理 & 計算DAG(Value Graph)基盤設計

- **スプリント期間**: 2026年11月25日 〜 (マイペース運用)
- **スプリントゴール**: 本コンペの学術的背景・引用文献(Addition Chain / SLP / Pebble Game / LNS)を体系化し、命令列を純粋な計算グラフ(Value Graph DAG)へ相互変換する基盤を設計・実装する。
- **役割**:
  - **PO (Product Owner)**: ユーザー様 (要件判断、スコープ決定、成果物レビュー)
  - **SM (Scrum Master)**: AI (スクラム規律の維持、学術調査、ブロッカー排除)
  - **Developers**: ユーザー様 ＋ AI (文献調査、アーキテクチャ設計、コード実装)

---

## 🎯 完成の定義 (Definition of Done: DoD)

1. [x] **学術的背景・学会・引用文献の体系化**:
   - 最短加算列(Addition Sequence)、直線プログラム合成(SLP)、ペブルゲーム理論、LNSに関する学会・重要論文の引用元を整理し、設計書ドキュメント化されていること。
2. [ ] **命令列 ⇄ Value Graph (DAG) 相互変換モジュール (`dag.py`) の実装**:
   - 既存の提出JSONプログラムから、レジスタ番号の依存を取り除いた純粋な有向非巡回グラフ(DAG)を構築できること。
   - 逆にDAGから、48レジスタに収まる合法な命令列を再構成できること。
3. [ ] **現在ポートフォリオ (128,219手) のDAG解析レポート**:
   - 1,000問の各プログラムにおける共有度(Sharing Rate)、プライベートチェーン長、レジスタ圧迫度(Peak Live Registers)を定量可視化すること。
4. [ ] **LNS (大規模近傍探索) のプロトタイプ設計**:
   - 次期スプリント(探索実行)に向けた、破壊(Destroy)と修復(Repair)の具体的アルゴリズム仕様が策定されていること。

---

## 📝 タスク一覧 & 進捗状況

| タスクID | 内容 | 担当 | 状態 | 成果物 / 備考 |
|---|---|:---:|:---:|---|
| **S001-1** | 学術的背景・関連学会・引用文献の調査整理 | AI+PO | Done | `002_Design/03_REFERENCES.md` |
| **S001-2** | 命令列 ➔ Value Graph (DAG) 変換器の実装 | AI | To Do | `workspace/src/dag.py` (Parser) |
| **S001-3** | Value Graph ➔ 48レジスタ命令列 再構成器の実装 | AI | To Do | `workspace/src/reg_alloc.py` (Linear Scan) |
| **S001-4** | 全1,000問のDAG構造・ボトルネック解析 | AI | To Do | `workspace/analyze_graph.py`, レポート |
| **S001-5** | LNS / Backward Beam エンジンの設計仕様書作成 | AI+PO | To Do | `002_Design/04_lns_engine_spec.md` |

---

## 📚 参照学術文献・知見ハイライト

詳細は **[002_Design/03_REFERENCES.md](../002_Design/03_REFERENCES.md)** を参照。

* **加算列の計算困難性**: Downey et al. (1981) - 複数ターゲットの最短加算列はNP完全。全探索ではなくメタヒューリスティクスが必須。
* **共有ハブの抽出**: Bos & Coster (1989) - 共通基底の発見と事前計算。
* **直線プログラム合成**: Gulwani et al. (POPL 2011) - SMT/SATによる局所最短プログラム合成。
* **レジスタ割り当て**: Hopcroft et al. (1977) のペブルゲーム理論、Poletto & Sarkar (1999) のLinear Scan。
* **LNS (大規模近傍探索)**: Shaw (CP 1998) - 部分破壊と修復による最短化ループ。

---

## 🔄 作業ログ (Work Log)

- **2026-11-25**:
  - Sprint 000 を完了 (初回提出 128,219手 / 341位)。
  - POからのレビューを受け、Sprint 001 を正式に計画・立案。
  - S001-1 に着手: 本コンペに関わる計算機科学・応用数学の学術文献、関連学会(STOC, SODA, CP, CPAIOR, POPL)、重要論文(Knuth, Yao, Downey, Bos-Coster, Gulwani, Shaw等)を徹底調査し、`002_Design/03_REFERENCES.md` に集約完了。
