# Santa 2026 - The Elf Calculator

Kaggleの年次最適化コンペティション「[Santa 2026 - The Elf Calculator](https://www.kaggle.com/competitions/santa-2026)」で **世界1位** を獲得するための研究開発リポジトリです。

---

## 🧭 プロジェクト全体マップ

```text
kgle_003_Santa2026/
├── 000_intro/                  # 大会規約・仕様・チュートリアル日本語ドキュメント
│   ├── 01_overview.md          # 大会概要、評価指標、スケジュール、賞金
│   ├── 02_data.md              # データ仕様、エルフの計算機ハードウェア仕様
│   ├── 03_code.md              # 公開コード動向、上位解法アプローチ
│   ├── 04_models.md            # 最適化モデル、ソルバー、探索手法
│   ├── 05_discussion.md        # コミュニティ議論、知見まとめ
│   └── 06_rules.md             # 公式大会規約 日本語全訳・解説
├── 001_Sprints/                # スプリントごとの詳細ログ (sprint_000.md ...)
├── 002_Design/                 # システム設計書 & 総合ナレッジベース
│   ├── 01_system_architecture.md # システムアーキテクチャ設計
│   └── 02_optimization_strategy.md # 最適化・探索アルゴリズム戦略
├── PROJECT_DASHBOARD.md        # 情報ラジエーター (現在地・ベストスコア・カンバン・PBL)
├── README.md                   # 本ファイル (全体案内)
│
└── workspace/                  # 開発環境ルート (VSCodeでここを開く)
    ├── .vscode/                # VSCode設定 (tasks.json 等)
    ├── src/                    # Python実装コード (検証器、ソルバー、探索器)
    └── output/                 # 生成されたプログラム成果物、提出ファイル
```

---

## 🚀 開発プロセス: Scrumban × Experiment-Driven

不定期な開発ペースと研究開発特有の「高い不確実性(10個試して9個失敗する探索タスク)」に対応するため、**スクラムバン(Scrumban)** をベースに **仮説・実験ドリブン管理(Experiment-Driven)** をドッキングさせたハイブリッドモデルを採用しています。

```text
【アジャイル開発(大きな理念)】
   │
   ├── 骨格・規律 ──▶ 【スクラム(Scrum)】
   │   ・3つの役割: PO (ユーザー様)、SM (AI)、Developers (ユーザー様＋AI)
   │   ・スプリントゴール、完成の定義(DoD)、KPT振り返り
   │
   ├── 運用の柔軟性 ──▶ 【カンバン(Kanban)】
   │   ・カレンダー日数で縛らない「マイペース運用」
   │   ・To Do / In Progress / Done のフロー管理
   │
   └── Kaggle最適化 ──▶ 【仮説・実験ドリブン (Experiment-Driven)】
       ・パズル群に対する高速な仮説検証ループ (Idea ➔ 実装 ➔ ローカル検証 ➔ 採用/却下)
       ・「Current Best Portfolio」による1,000問別ベストプログラムの常時維持
       ・ベスト更新時のみ自動合体パッケージングしてKaggleへ提出
```

### 2段階のフェーズ運用

1. **Phase 1: 基盤構築フェーズ (Sprint 000)**
   - パイプラインやローカル検証器(Fast Zero-Trust Validator)をゼロから実装・整備する純粋なエンジニアリング作業。スクラムバン(WBS・カンバン)で着実に進行。
2. **Phase 2: スコア改善・探索フェーズ (Sprint 001以降)**
   - 重複管理を排除し、**[PROJECT_DASHBOARD.md](./PROJECT_DASHBOARD.md)** のトップに置いた「**Current Best Portfolio**」をベースにした高速な仮説検証ループへシフト。
   - パズルファミリー別/パズル個別に実験を行い、ローカル検証でステップ数削減が確認された場合のみポートフォリオを更新して提出。

```text
【探索フェーズの実験ループ (1実験 ＝ 数分〜数時間)】
  仮説立案 (Idea) ➔ コード変更 ➔ verify_all.pyで採点
   ├── スコア悪化 / 制約違反 ➔ 却下理由を記録してコード破棄
   └── スコア更新! ➔ ベストプログラムを更新 ➔ 自動マージしてsubmit!
```

---

## 📊 主要リンク
- **[PROJECT_DASHBOARD.md](./PROJECT_DASHBOARD.md)**: 最新ベストスコア、スプリントカンバン、バックログ(PBL)
- **[000_intro/README.md](./000_intro/README.md)**: 大会仕様・ルール日本語まとめ
- **[001_Sprints/sprint_000.md](./001_Sprints/sprint_000.md)**: 現在進行中のSprint 000ログ
