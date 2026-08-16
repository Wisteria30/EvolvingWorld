# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## リポジトリ概要

論文「EvolvingWorld: An Open-Schema Framework for Co-Evolving Role-Play Agents and World Model in Interactive Literary World」の公式リポジトリ。書籍からの構造化データ抽出 → SFT学習 → 2モデルによるシーン単位シミュレーション → LLM-as-a-Judge評価、という研究パイプライン一式を含む。テストスイート・リンターは存在しない。非学習系のPython環境はルートの `pyproject.toml` と `uv.lock` で管理する。

## 実行の前提

- **すべてのコマンドはリポジトリルート `EvolvingWorld/` から実行する。** `data_construction/utils.py` と `evaluation/main.py` は `config.json` を相対パスで直接openするため、cwdがルートでないと失敗する。
- ルートの `config.json`(`{"api_key": ..., "base_url": ...}`)がOpenAI互換APIの接続設定。リモートAPIを使う全スクリプトが参照する。
- データ構築・シミュレーション・評価は `uv sync` で環境を構築し、`uv run` 経由で実行する。vLLMはプラットフォーム依存のためルート環境には含めない。LLaMA-Factory学習環境は従来どおり別管理。
- データセットは `uv run hf download zongqing0068/EvolvingWorld --repo-type dataset --local-dir dataset` で取得(`dataset/` はgit管理外)。

## 主要コマンド

```bash
# シミュレーション(リモートAPI、1サンプルのみ)
uv run python simulation/main.py --input dataset/test/test_all.json --mode remote \
  --world-model gemini-2.5-pro --character-agent-model gemini-2.5-pro \
  --offset 0 --limit 1 --output-dir simulation/outputs/example_run

# シミュレーション(ローカルvLLM。事前にworld用/character用の2サーバーを起動)
uv run python simulation/main.py --input dataset/test/test_all.json --mode local \
  --world-base-url http://127.0.0.1:8000/v1 \
  --character-agent-base-url http://127.0.0.1:8001/v1 --max-tokens 8192

# 失敗サンプルの再実行(--rerun は --offset/--limit を無視して欠損・途中終了分を走査)
uv run python simulation/main.py ... --rerun --num-workers 4 --output-dir <same_dir>

# 評価(<run_name> は simulation/outputs/ 配下のディレクトリ名)
uv run bash evaluation/run_all_eval.sh --judge gemini-2.5-pro <run_name>
# 単体実行・特定サンプルのみ
uv run python evaluation/main.py --input_dir simulation/outputs/<run_name> \
  --input_snapshots dataset/test/test_all.json --samples 0,1,2

# 学習(model_a=世界モデル、model_b=キャラクターエージェント。個別に学習)
bash training/run.sh --model model_a --mode full --gpu 0 --model-name-or-path Qwen/Qwen2.5-7B-Instruct

# データ構築(ステップ1はLLM呼び出しが大量。デバッグ時は小さいJSONLで)
uv run python data_construction/main.py --input dataset/original_books_from_gutenberg.jsonl \
  --output_dir data --num_workers 8 --model gemini-2.5-pro --candidate_model claude-sonnet-4-5
uv run python data_construction/transform.py --dir dataset/extracted_data --seed 40
```

## アーキテクチャ

### 2モデル分業(コード全体を貫く命名規則)

- **model_a = 世界モデル(director)**: `scene_cast`(次シーンの出演者決定)、`location_scenario`(場所とシナリオ)、`next_character`(次の話者選択/シーン終了判定)、`world_update`(世界全体・場所単位の状態更新)
- **model_b = キャラクターエージェント(actor)**: `interaction_gen`(インタラクション生成)、`character_update`(プロフィール・hidden tracker更新)、`motivation_update`(次シーンの動機生成)

このタスク名7種が、学習データファイル名(`dataset/train/model_{a,b}_<task>.json`)、シミュレーションのtrace、評価のタスク→指標マッピングまで一貫して使われる。

### データフロー(モジュール間の受け渡し)

```
生書籍JSONL → data_construction/main.py → dataset/extracted_data/{scenes,character_dynamic,world_dynamic}/
           → data_construction/transform.py → dataset/train/(ShareGPT形式) + dataset/test/(スナップショット) + dataset/book_split.json
dataset/train/ → training/run.sh(LLaMA-Factory SFT)
dataset/test/test_all.json → simulation/main.py → simulation/outputs/<run>/sample_NNNNNN/
simulation/outputs/<run>/ + test_all.json → evaluation/ → evaluation/results/<run>_<judge>_final/
```

- 書籍分割は `transform.py` が実施: 10%がOODテスト書籍、残りの半分がtrain+test(前70%学習・後30%がIDテスト)、残り半分は全シーン学習用。
- シミュレーション出力 `sample_NNNNNN/` の各ファイル(`meta.json` / `all_scenes.json` / `character_dynamic.json` / `world_dynamic.json` / `trace.json`)は各ターン後にリアルタイム書き込みされるため、クラッシュしても途中結果が残る。
- シーンindexの対応: 生成シーン `k` は原作の `source_scene_index + k` に対応(`all_scenes.json` 内の `scene_index` はラン内相対で0始まり)。

### シミュレーションループ(simulation/simulator.py の StorySimulator)

シーン計画(scene_cast → location_scenario → motivation生成)→ シーン実行(next_character → interaction_gen → world_update のループ、最大 `--max-turns-per-scene`)→ 状態リフレクション(character_update)を `--max-scenes` 回繰り返す。プロンプト構築もすべて `simulator.py` 内。`inference.py` はOpenAI互換クライアントで、モデル名からreasoningモデル/Qwen3系を判別してパラメーター形式を切り替える。`utils.py` に切り詰められたJSONの修復・他キャラのthoughtマスキングなどのヘルパーがある。モデルの出力パース失敗はシミュレーションを `stop_reason='error'` で終了させ、評価時に責任モデルのIC指標へ対数減衰ペナルティが課される。

### 評価(evaluation/)

LLM-as-a-Judge。CHARACTER(6次元11指標)+ WORLD(4次元9指標)の計20指標を、基礎点50 + merit/demerit 加減点(0–100)で採点。`main.py` がサンプル並列で `judge.py` を呼び、`aggregator.py` が集計。指標定義とタスク→指標マッピングは `evaluation/README.md` が正典。

## 詳細ドキュメント

各モジュールのREADMEにパラメーター表と手順の詳細がある: `data_construction/README.md`、`simulation/README.md`、`evaluation/README.md`、`training/README.md`。ルートREADMEは日本語。
