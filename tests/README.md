# `tests` Pythonファイル学習ガイド

このディレクトリには、銘柄選択、データ取得、特徴量、scaler、因子、LSTM、生成、ONNX変換を確認するテストがあります。

## テストの実行

```bash
pytest
```

テストでは、できるだけ合成データやモックを使用し、ネットワーク接続や大量の学習データに依存しないようにしています。

## データ準備に関するテスト

| ファイル | 確認内容 |
|---|---|
| [`test_current_master.py`](test_current_master.py) | 同梱の銘柄マスターに11セクターが存在し、銘柄コードが重複していないことを確認します。 |
| [`test_metadata.py`](test_metadata.py) | 偽のYahoo情報を使い、会社名、セクター、時価総額、情報源、取得日が補完されることを確認します。 |
| [`test_universe.py`](test_universe.py) | ETF、REIT、未知の上場情報などが除外され、時価総額順に銘柄が選ばれることを確認します。 |
| [`test_fetch_data.py`](test_fetch_data.py) | ダウンロード処理をモック化し、キャッシュ、欠損処理、TOPIX代理系列を確認します。 |

## 特徴量・前処理のテスト

| ファイル | 確認内容 |
|---|---|
| [`test_features.py`](test_features.py) | 合成OHLCVから特徴量が作成され、NaN・無限大・未来情報混入がないことを確認します。 |
| [`test_preprocessing.py`](test_preprocessing.py) | `FeatureScaler`の標準化と逆変換が正しく、元の値へ戻ることを確認します。 |
| [`test_factors.py`](test_factors.py) | `market`、`rotation`、`volatility`因子、loadings、共通ノイズ重みの形状と範囲を確認します。 |

## モデル・生成のテスト

| ファイル | 確認内容 |
|---|---|
| [`test_model.py`](test_model.py) | LSTMが標準化済みOHLCVを受け、`[batch, 11, 5]`を返すことと損失を確認します。 |
| [`test_generation.py`](test_generation.py) | OHLCV列、OHLC制約、ローソク足画像、イベント互換処理を確認します。 |
| [`test_evaluate.py`](test_evaluate.py) | 評価画像がclose経路だけを表示し、旧ボラティリティ・リターン画像を作らないことを確認します。 |
| [`test_export_onnx.py`](test_export_onnx.py) | ONNXの入出力形状、LSTM演算子、OHLCV scaler metadataを確認します。 |

## 学習のポイント

テストは、コードが動くかだけでなく、数式やデータの意味が正しいかを確認するためにあります。特に次のテストは、今回の株価生成モデルで重要です。

- scalerの変換と逆変換
- 未来データの混入防止
- OHLCVの標準化と逆変換
- `High`、`Low`を含むOHLCの大小関係
- ONNXへ渡すOHLCV列順の一致
