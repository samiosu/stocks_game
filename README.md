# 日本株セクター市場生成モデル

日本株の過去のOHLCVから11セクターの次のOHLCVをLSTMで生成し、ゲーム用の市場系列として出力するPython実装です。売買シグナルや将来価格の予測を目的としません。

## 構成

処理は次のCLIで段階的に実行できます。

```bash
python -m pip install -e '.[reports,onnx,dev]' && \
python -m stock_sim.select_universe --as-of 2026-08-18 && \
python -m stock_sim.fetch_data --start 2015-01-01 && \
python -m stock_sim.build_features && \
python -m stock_sim.train --config config/config.yaml && \
python -m stock_sim.generate --days 120 --seed 42 && \
python -m stock_sim.evaluate && \
python -m stock_sim.export_onnx
```

`python -m ...` が使える環境で実行してください。`python` がない環境では `python3` に読み替えます。

同梱の `data/metadata/stock_master.csv` には、現時点で使用する固定33銘柄を登録しています。社名変更・銘柄コード変更の追跡は行わず、現行ティッカーで2015年以降の価格履歴を取得します。時価総額は実行時にYahoo Financeから取得してセクター内順位を付けます。前段が失敗した場合に後続を実行しないため、上記のように `&&` を使っています。

## 銘柄マスターと分類

`data/metadata/stock_master.csv` は現行ティッカーを固定する入力ファイルです。初期状態で33銘柄を登録しており、必要な場合だけこのCSVを差し替えます。過去の社名変更・銘柄コード変更を追跡するための仕組みは使用しません。

JPX CSVから別の現行銘柄リストを作る場合は、補助コマンドで列名を正規化できます。正規化後に継続上場確認と時価総額を補完してください。

JPXは上場銘柄一覧と東証上場日をJPX Data PortalからCSV取得できると案内しています。[東証上場銘柄一覧](https://www.jpx.co.jp/markets/statistics-equities/misc/01.html)

```bash
python -m stock_sim.metadata --jpx-input jpx_stock_list.csv --output data/metadata/stock_master.csv
```

セクターは固定マスターの `sector_id` を優先し、手動上書き `data/sector_overrides.csv`、Yahoo Finance/GICS相当の名称の順に解決します。解決できない銘柄、上場日不明、継続上場未確認の銘柄は採用せず、`data/metadata/universe_rejections.csv` に理由を記録します。

採用条件は、2015-01-01以前に東証へ上場し、2015年以降の上場廃止期間がなく、現時点（`--as-of` または実行日）の時価総額がセクター内で大きい普通株式です。ETF、ETN、投資信託、REIT、インフラファンド、優先株、外国企業は除外します。各セクター最大3銘柄で、3銘柄に満たないセクターは補充しません。

現在の時価総額上位を使って2015年以降を学習するため、survivorship biasがあります。これはゲーム用の市場生成という目的のための意図した仕様です。

## データと注意事項

価格取得はまず `yfinance` を使い、選定銘柄を一括取得します。`data/raw/prices.parquet` と `data/raw/indices.parquet` に保存し、取得済み範囲はキャッシュとして再利用します。失敗・欠損・期間不足はログに記録し、重複行は除去します。

Yahoo Financeで `^TPX` の履歴が欠損する場合は、設定済みの `1306.T`（TOPIX連動ETF）をTOPIXの代理系列として取得し、内部では `^TPX` として保存します。この場合はログに警告を出します。

`yfinance` は非公式データソースです。データ完全性は保証されず、Yahoo Finance側の仕様変更で取得できなくなる可能性があります。商用利用や再配布を行う場合は、Yahoo Financeおよび関連サービスの利用規約を確認してください。Yahooのメタデータ解決結果は `data/metadata/metadata_cache.parquet` にも保存します。

## 特徴量と学習

`sector_data.parquet` には従来のリターン・ボラティリティ等の分析用特徴量に加え、セクターごとの生OHLCVを保持します。LSTMの学習入力はこのうち `ohlcv_columns`（11セクター×Open/High/Low/Close/Volumeの55列）です。セクター系列はデフォルトで構成銘柄の等ウェイト平均です。`config/config.yaml` の `sector_aggregation.method` を `market_cap_weighted` にすると時価総額ウェイトへ変更できます。個別銘柄特徴量は `data/processed/stock_features.parquet` に保持します。

入力系列は60営業日、LSTMの出力ヘッドは `[batch, 11, 5]`（ギャップ、実体、上下ヒゲ、直近出来高からの比率）です。学習時は生OHLCVの入力を標準化し、相対ターゲットを学習します。生成時は前バーの終値と直近20日出来高の幾何平均から生OHLCVへ再構成するため、価格水準の退行、長いヒゲ、出来高の累積暴走を抑えられます。5日間のteacher-forcing rollout lossと、検証残差に基づく確率ノイズを使い、出来高ノイズは価格より小さくします。`high >= max(open, close)`、`low <= min(open, close)`、各値の正値制約は再構成時に保証します。

モデルはUnity Sentis/Inference Engineで対応しているONNX `LSTM` 演算子を使います。GRU時代のチェックポイントはLSTMと重み形式が異なるため再利用せず、設定を更新した状態で学習とONNX出力をやり直してください。

学習・検証・テストは時系列順に固定し、デフォルトは2015–2021、2022–2023、2024–実行時点です。スケーラーはOHLCVの学習期間だけでfitし、`models/feature_scaler.pkl` に保存します。GRUや旧確率的LSTMのチェックポイントは互換性がないため、OHLCV設定で再学習してください。

## 生成とUnity

生成は過去60行のOHLCVから相対値を自己回帰予測し、前バーの終値と直近20日出来高の幾何平均を基準に次のOHLCVを再構成します。Python生成はseedと検証残差ノイズを使うため、同じseedなら同じ系列を再現します。結果は `data/generated/generated_prices.parquet`（close互換列＋55個のOHLCV列）、`data/generated/generated_ohlcv.parquet`（OHLCV専用）、互換用closeリターンは `reports/raw_generated_returns.csv`、ローソク足画像は `reports/figures/generated_candlestick.png` に保存されます。

`export_onnx` は `models/lstm_model.onnx` と `models/lstm_model.metadata.json` を作成します。Unity Sentis/Inference Engineでは、float32の標準化済み `[1, 60, 55]` を `features` 入力へ渡し、`ohlcv` 出力 `[1, 11, 5]` を `open/high/low/close/volume` の順で受け取ります。入力標準化の列順・平均・標準偏差と、内部相対表現の仕様はmetadata JSONに保存されます。

Unityへの導入手順、Unity用メタデータの作成、C#実装例、複数日生成の注意点は [Unity統合マニュアル](docs/UNITY_INTEGRATION.md) を参照してください。

Pythonファイルの役割を学習用にまとめた説明は、[`src/stock_sim/README.md`](src/stock_sim/README.md)と[`tests/README.md`](tests/README.md)を参照してください。ディレクトリごとに、含まれるPythonファイルを一覧化しています。

## テスト

```bash
pytest
```

テストはネットワークに依存せず、ユニバース選定、未来情報混入防止、NaN/inf、再現可能な生成、（依存環境にある場合）ONNX入出力を確認します。
