# `src/stock_sim` Pythonファイル学習ガイド

このディレクトリには、日本株のデータを取得し、11セクターの値動きを学習・生成・評価し、Unity用ONNXへ変換するPythonモジュールが入っています。

## 全体の処理順

```text
metadata.py / universe.py
        ↓
fetch_data.py
        ↓
features.py
        ↓
preprocessing.py / dataset.py
        ↓
factors.py / model.py / train.py
        ↓
generate.py
        ↓
evaluate.py
        ↓
export_onnx.py
```

## 共通処理

| ファイル | 役割 | 学習ポイント |
|---|---|---|
| [`__init__.py`](__init__.py) | `stock_sim`をPythonパッケージとして認識させます。 | 処理がなくても、`python -m stock_sim.generate`のようなモジュール実行に必要です。 |
| [`constants.py`](constants.py) | 11セクター、イベント、特徴量名などの共通定数を定義します。 | セクター順と列名を全処理で統一します。 |
| [`config.py`](config.py) | YAML設定、パス、日付、ログ、乱数seedを扱います。 | 学習条件をコードから分離し、同じseedで再現しやすくします。 |
| [`io.py`](io.py) | Parquetの読み書きを共通化します。 | データ入出力を一箇所にまとめています。 |

## 銘柄とデータ取得

| ファイル | 役割 | 主な入出力 |
|---|---|---|
| [`metadata.py`](metadata.py) | 銘柄コード、会社名、セクター、上場日、時価総額などを正規化します。JPX CSVの変換やYahoo Financeのメタデータ補完も行います。 | `data/metadata/stock_master.csv`、メタデータキャッシュ |
| [`universe.py`](universe.py) | ETF、REIT、外国企業などを除外し、セクター内の時価総額順位から使用銘柄を選びます。 | `data/selected_universe.csv`、除外理由CSV |
| [`select_universe.py`](select_universe.py) | `universe.py`をCLIから実行する入口です。 | `python -m stock_sim.select_universe` |
| [`fetch_data.py`](fetch_data.py) | Yahoo Financeから個別銘柄と日経平均・TOPIXのOHLCVを取得し、キャッシュします。 | `data/raw/prices.parquet`、`data/raw/indices.parquet` |

`fetch_data.py`では、TOPIXの`^TPX`が不完全な場合に`1306.T`を代理系列として使用します。外部APIの失敗、欠損、重複、期間不足も確認します。

## 特徴量とデータセット

| ファイル | 役割 | 学習ポイント |
|---|---|---|
| [`features.py`](features.py) | OHLCVから対数リターン、出来高変化、5日・20日ボラティリティ、指数との差などを計算し、セクター系列を作ります。 | 価格水準ではなく、`log(close_t / close_t-1)`を中心に学習します。未来のデータを使わないことが重要です。 |
| [`build_features.py`](build_features.py) | `features.py`の処理をCLIから実行するラッパーです。 | 実際の計算は`features.py`にあります。 |
| [`preprocessing.py`](preprocessing.py) | 特徴量を標準化し、平均・標準偏差の保存と逆変換を行います。 | scalerは学習期間だけでfitし、生成時も同じ値を使います。 |
| [`dataset.py`](dataset.py) | 過去60期間などの入力ウィンドウと、予測対象のリターン・ボラティリティを作ります。 | 入力範囲に未来のデータが混ざらないようにします。 |

主な中間ファイルは次の通りです。

- `data/processed/stock_features.parquet`
- `data/processed/sector_data.parquet`
- `data/processed/feature_schema.json`
- `models/feature_scaler.pkl`

## 因子モデルとGRU

| ファイル | 役割 | 学習ポイント |
|---|---|---|
| [`factors.py`](factors.py) | 11セクターのリターンから`market`、`rotation`、`volatility`の共通因子とloadingsを推定します。 | セクターを独立乱数にせず、実データに近い相関構造を作ります。 |
| [`model.py`](model.py) | 過去の特徴量から、各セクターの次のリターン分布の`mu`と`log_sigma`を出力するGRUです。 | 点予測ではなく、`sigma = softplus(log_sigma) + 1e-5`で連続的な確率分布を生成します。 |
| [`train.py`](train.py) | 時系列分割、scaler、因子推定、GRU学習、早期停止、チェックポイント保存を行います。 | モデル重みだけでなく、列順、scaler、因子情報も保存します。 |

モデルの主な入出力は次の形です。

```text
入力 : [batch_size, 60, feature_size]
出力 : [batch_size, 11, 2]
       └─ mu, log_sigma
```

## 生成・評価・Unity出力

| ファイル | 役割 | 主な出力 |
|---|---|---|
| [`generate.py`](generate.py) | GRUからリターンをサンプリングし、価格とOHLCを生成します。 | `generated_prices.parquet`、`generated_ohlc.parquet`、ローソク足PNG |
| [`evaluate.py`](evaluate.py) | 実データと生成データのリターン、ボラティリティ、自己相関、ドローダウン、セクター相関などを比較します。 | CSV、PNG、`evaluation_report.html` |
| [`export_onnx.py`](export_onnx.py) | PyTorchモデルをUnity向けONNXへ変換し、scalerや因子情報をJSONに保存します。 | `models/gru_model.onnx`、`models/gru_model.metadata.json` |

`generate.py`の価格再構成は次の式です。

```python
price_next = price_current * exp(return_next)
```

OHLCは、生成されたClose-to-CloseリターンからOpen・High・Lowを連続乱数で構成します。`High >= max(Open, Close)`、`Low <= min(Open, Close)`などのローソク足制約も確認します。

ONNXにはモデル推論部分が入ります。特徴量作成、scaler、乱数サンプリング、価格再構成、OHLC生成はUnity側でも同じ仕様を使う必要があります。

## CLI実行順

```bash
python -m stock_sim.select_universe
python -m stock_sim.fetch_data
python -m stock_sim.build_features
python -m stock_sim.train --config config/config.yaml
python -m stock_sim.generate --days 120 --seed 42
python -m stock_sim.evaluate
python -m stock_sim.export_onnx
```

