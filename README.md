[![make.fun Bounty](https://make.fun/badge/news-image-20261126.svg)](https://make.fun/bounty/news-image-20261126)
[![make.fun Rank](https://make.fun/badge/news-image-20261126/mugi732/news-mosaic-20261126/rank.svg)](https://make.fun/bounty/news-image-20261126/leaderboard)
[![make.fun Reward](https://make.fun/badge/news-image-20261126/mugi732/news-mosaic-20261126/reward.svg)](https://make.fun/bounty/news-image-20261126)

# World News Mosaic (2026-11-26)

2026年11月26日の国際ニュース30件を1枚の1024×1024 PNG画像（`entry.png`）にまとめた提出用リポジトリです。

## ファイル構成

- `main.py` — 提出用エントリポイント（Python 標準ライブラリのみで動作）。同ディレクトリ内の `entry.png` を出力先ディレクトリへコピーし、SHA-256 ハッシュを表示します。
- `entry.png` — 生成済みの提出画像（1024×1024 PNG）。
- `generate_news_mosaic_20261126.py` — `news_20261126.md` から `entry.png` を生成するスクリプト（実行には `numpy` と `Pillow` が必要）。
- `news_20261126.md` — 画像生成の入力となるニュース一覧テキスト。

## 使い方

### 提出プログラムの実行

環境変数 `MAKEFUN_OUTPUT_DIR` が設定されている場合はそのディレクトリへ、未設定の場合は `/output` へ `entry.png` を出力します。

```bash
MAKEFUN_OUTPUT_DIR=out python3 main.py
```

### 画像の再生成（任意）

ローカル環境で画像を再生成する場合は、`numpy` および `Pillow` をインストールした上で以下を実行します。

```bash
python3 generate_news_mosaic_20261126.py
```

## ライセンス

MIT License（詳細は `LICENSE` を参照してください）。
