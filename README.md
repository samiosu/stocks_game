# 日本株セクター市場生成モデル

日本株の過去の値動きから、市場全体の局面・セクターローテーション・変化するボラティリティ・イベント反応を学習し、ゲーム用のセクター価格系列を確率的に生成するPython実装です。売買シグナルや将来価格の予測を目的としません。

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

株価水準ではなく、対数リターン、OHLC相対値、出来高変化、5日/20日ボラティリティ、TOPIX・日経平均に対する超過リターンを使います。セクター系列はデフォルトで構成銘柄の等ウェイト平均です。`config/config.yaml` の `sector_aggregation.method` を `market_cap_weighted` にすると時価総額ウェイトへ変更できます。個別銘柄特徴量は `data/processed/stock_features.parquet` に保持します。

入力系列は60営業日、GRUの出力は `[batch, 11, 2]`（各セクターの `mu` と `log_sigma`）です。`sigma = softplus(log_sigma) + 1e-5` として、実データから推定したmarket/rotation/volatilityのfactor loadingを使った共通ノイズと固有ノイズからリターンをサンプリングします。価格は `close[t+1] = close[t] * exp(return[t+1])` で積み上げます。イベント入力がない場合は全イベント要素が0です。`horizon` は1または5に切り替えられます。

学習・検証・テストは時系列順に固定し、デフォルトは2015–2021、2022–2023、2024–実行時点です。スケーラーは学習期間だけでfitし、`models/feature_scaler.pkl` に保存します。factor loadingはゲーム用の相関構造として全履歴から推定し、checkpoint内の `factor_metadata` に保存します。

## 生成とUnity

生成にはseed、日数、初期価格、イベントスケジュール、ボラティリティ倍率、連続soft clipを指定できます。百分位hard clipは必要な場合だけ明示的に有効化します。同じseed・同じチェックポイント・同じ入力履歴なら同じ系列を再現します。close系列から始値ギャップと日中レンジを連続乱数で生成し、OHLC制約を満たすローソク足を作ります。結果は `data/generated/generated_prices.parquet`（close互換列＋OHLC列）、`data/generated/generated_ohlc.parquet`（OHLC専用）、raw sampled returnsは `reports/raw_generated_returns.csv`、ローソク足画像は `reports/figures/generated_candlestick.png` に保存されます。

`export_onnx` は `models/gru_model.onnx` と `models/gru_model.metadata.json` を作成します。Unity Sentis/Inference Engineでは、float32の `[1, 60, feature_size]` を `features` 入力へ渡し、`params` 出力 `[1, 11, 2]` の最後の次元をmu/log_sigmaとして利用します。ONNXはモデルの推論部だけを含み、特徴量計算・スケーリング・イベントベクトル生成・乱数サンプリングはPythonまたはUnity側で同じ仕様を実装してください。特徴量の順序、scaler、factor loading、生成設定はmetadata JSONに保存されます。

Unityへの導入手順、Unity用メタデータの作成、C#実装例、複数日生成の注意点は [Unity統合マニュアル](docs/UNITY_INTEGRATION.md) を参照してください。

Pythonファイルの役割を学習用にまとめた説明は、[`src/stock_sim/README.md`](src/stock_sim/README.md)と[`tests/README.md`](tests/README.md)を参照してください。ディレクトリごとに、含まれるPythonファイルを一覧化しています。

## テスト

```bash
pytest
```

テストはネットワークに依存せず、ユニバース選定、未来情報混入防止、NaN/inf、再現可能な生成、（依存環境にある場合）ONNX入出力を確認します。
