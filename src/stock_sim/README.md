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
| [`features.py`](features.py) | OHLCVから分析用特徴量を計算し、同時にセクター別の生OHLCV列を作ります。 | LSTMの主入力は`ohlcv_columns`の55列です。分析用のリターン・ボラティリティ列は学習入力ではありません。 |
| [`build_features.py`](build_features.py) | `features.py`の処理をCLIから実行するラッパーです。 | 実際の計算は`features.py`にあります。 |
| [`preprocessing.py`](preprocessing.py) | 特徴量を標準化し、平均・標準偏差の保存と逆変換を行います。 | scalerは学習期間だけでfitし、生成時も同じ値を使います。 |
| [`dataset.py`](dataset.py) | 過去60期間のOHLCV入力ウィンドウと、次のOHLCV targetを作ります。 | 入力範囲に未来のデータが混ざらないようにします。旧リターン用関数も評価互換のため残しています。 |

主な中間ファイルは次の通りです。

- `data/processed/stock_features.parquet`
- `data/processed/sector_data.parquet`
- `data/processed/feature_schema.json`
- `models/feature_scaler.pkl`

## OHLCVモデルとLSTM

| ファイル | 役割 | 学習ポイント |
|---|---|---|
| [`model.py`](model.py) | 過去の標準化済みOHLCVから、相対OHLCVダイナミクスを出力するLSTMです。 | 入力・出力とも11セクター×5項目で、出力形状は`[batch, 11, 5]`です。 |
| [`train.py`](train.py) | 時系列分割、入力・相対ターゲットscaler、5日rollout LSTM学習、早期停止、チェックポイント保存を行います。 | 検証残差から生成ノイズも校正します。 |

モデルの主な入出力は次の形です。

```text
入力 : [batch_size, 60, 55]
       └─ 11セクター × (open, high, low, close, volume)
出力 : [batch_size, 11, 5]
       └─ (gap, body, upper_wick, lower_wick, log_volume_ratio)
再構成 : 前バーのcloseと直近20日出来高の幾何平均を基準に次の生OHLCV
```

## 生成・評価・Unity出力

| ファイル | 役割 | 主な出力 |
|---|---|---|
| [`generate.py`](generate.py) | 相対値をLSTMで予測し、前バーから次のOHLCVを自己回帰再構成します。 | `generated_prices.parquet`、`generated_ohlcv.parquet`、ローソク足PNG |
| [`evaluate.py`](evaluate.py) | 実データと生成データの数値指標を比較します。画像はclose経路比較だけを出力し、旧ボラティリティ・リターン分布・相関画像は作りません。 | CSV、close経路PNG、`evaluation_report.html` |
| [`export_onnx.py`](export_onnx.py) | PyTorchモデルをUnity向けONNXへ変換し、OHLCV scalerをJSONに保存します。 | `models/lstm_model.onnx`、`models/lstm_model.metadata.json` |

`generate.py`は、標準化された相対出力を次のように生OHLCVへ戻します。

```python
open = previous_close * exp(gap_log_return)
close = open * exp(body_log_return)
high = max(open, close) * exp(upper_wick_log_range)
low = min(open, close) * exp(-lower_wick_log_range)
volume_reference = exp(0.9 * log(trailing_20d_geometric_mean_volume) + 0.1 * log(training_volume_anchor))
volume = clip(volume_reference * exp(log_volume_ratio), previous_volume / 3, previous_volume * 3)
```

生成後は`High >= max(Open, Close)`、`Low <= min(Open, Close)`、Open/Close/Volumeの正値制約を確認します。

`inference.py`が学習rollout・Python・ONNXに共通の復元処理です。`model.py`内で入力をウィンドウ相対の対数値へ正規化します。Unityの入力はmetadataのscalerで標準化したOHLCV、出力 `[1, 11, 5]` は生OHLCVです。通常ONNXはノイズなし、追加の `lstm_model.stochastic.onnx` は明示的な `residual [1,11,5]` 入力を受け取ります。`audit.py`は固定シナリオで分布と長期挙動を記録します。

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
