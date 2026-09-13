# OHLCV LSTMモデルをUnityで使用する

このプロジェクトのONNXモデルは、GRUではなくUnityで扱えるLSTMを使用します。過去60営業日の11セクター分OHLCVを入力し、内部では相対OHLCVダイナミクスを推定します。ONNXの外部出力は、前バーの終値と直近20日出来高の幾何平均から再構成済みの次の11セクター分の生OHLCVです。

## 1. 学習とONNX出力

```bash
python -m stock_sim.features --config config/config.yaml
python -m stock_sim.train --config config/config.yaml
python -m stock_sim.export_onnx --config config/config.yaml
```

旧`gru_model.*`や旧確率的LSTMのチェックポイントは、このモデルとは入出力形状が異なるため使用しません。

## 2. Unityへコピーするファイル

| ファイル | 用途 |
|---|---|
| `models/lstm_model.onnx` | ノイズなしのOHLCV推論 |
| `models/lstm_model.stochastic.onnx` | `residual [1,11,5]` を追加で受け取る確率的生成用モデル |
| `models/lstm_model.metadata.json` | 入出力名、列順、scaler、OHLCV仕様 |
| `data/processed/initial_ohlcv_window.json` | 初回推論用の過去60行 |
| `data/generated/generated_ohlcv.parquet` | Python側で生成したOHLCVの確認用 |

`models/lstm_model.pt`と`models/feature_scaler.pkl`はPython側の学習用です。Unityではmetadata JSONに出力された`mean`と`scale`を使います。

## 3. 入出力仕様

実測仕様は次のとおりです。

| 項目 | 値 |
|---|---|
| 入力名 | `features` |
| 入力形状 | `[1, 60, 55]` |
| 出力名 | `ohlcv` |
| 出力形状 | `[1, 11, 5]` |
| 項目順 | `open, high, low, close, volume` |
| セクター順 | `energy, materials, industrials, consumer_discretionary, consumer_staples, health_care, financials, information_technology, communication_services, utilities, real_estate` |

55列の順序はmetadataの`ohlcvColumns`を唯一の正解とします。現在は次のセクターmajor順です。

```text
energy__open, energy__high, energy__low, energy__close, energy__volume,
materials__open, ...,
real_estate__open, real_estate__high, real_estate__low, real_estate__close, real_estate__volume
```

ONNXへ渡す前に、各列を次で標準化します。

```text
normalized[i] = (raw[i] - mean[i]) / scale[i]
```

ONNX出力はすでに生OHLCVです。`output`はセクターmajorの`[open, high, low, close, volume]`です。内部で相対ターゲットscalerの逆変換と、次の式による再構成を行います。

```text
open = previousClose * exp(gap)
close = open * exp(body)
high = max(open, close) * exp(upperWick)
low = min(open, close) * exp(-lowerWick)
volumeReference = exp(0.9 * log(trailing20dGeometricMeanVolume) + 0.1 * log(trainingVolumeAnchor))
volume = clamp(volumeReference * exp(logVolumeRatio), previousVolume / 3, previousVolume * 3)
```

出力は`open`・`close`・`volume`が正値で、`high >= max(open, close)`、`low <= min(open, close)`となるようグラフ内で制約されます。

## 4. 初期ウィンドウの作成

初回入力は、生OHLCVの過去60行です。Python側でJSONを作る場合は次を実行します。

```bash
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

schema = json.loads(Path("data/processed/feature_schema.json").read_text(encoding="utf-8"))
columns = schema["ohlcv_columns"]
frame = pd.read_parquet("data/processed/sector_data.parquet").sort_values("date").tail(60)
if len(frame) != 60:
    raise RuntimeError(f"60 rows are required, got {len(frame)}")
payload = {
    "sequenceLength": 60,
    "featureSize": len(columns),
    "ohlcvColumns": columns,
    "values": frame[columns].to_numpy(dtype="float32").reshape(-1).tolist(),
}
Path("data/processed/initial_ohlcv_window.json").write_text(
    json.dumps(payload), encoding="utf-8"
)
PY
```

この`values`は標準化前です。Unity側でmetadataの`mean`と`scale`を使って標準化します。

## 5. C#最小実装例

`Assets/Scripts/StockMarketOnnxRunner.cs`の例です。Unity 6の`Unity.InferenceEngine` APIを想定しています。

```csharp
using System;
using Unity.InferenceEngine;
using UnityEngine;

public sealed class StockMarketOnnxRunner : MonoBehaviour
{
    [Serializable]
    private sealed class Metadata
    {
        public int sequenceLength;
        public int featureSize;
        public string[] ohlcvFields;
        public string[] ohlcvColumns;
        public Scaler scaler;
    }

    [Serializable]
    private sealed class Scaler
    {
        public float[] mean;
        public float[] scale;
    }

    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private TextAsset metadataJson;
    [SerializeField] private BackendType backend = BackendType.CPU;

    private const int SectorCount = 11;
    private const int FieldCount = 5;
    private Worker worker;
    private Metadata metadata;

    private void Awake()
    {
        metadata = JsonUtility.FromJson<Metadata>(metadataJson.text);
        if (metadata.sequenceLength != 60 || metadata.featureSize != SectorCount * FieldCount)
            throw new InvalidOperationException("Expected OHLCV input shape [1, 60, 55].");
        if (metadata.scaler == null || metadata.scaler.mean.Length != metadata.featureSize || metadata.scaler.scale.Length != metadata.featureSize)
            throw new InvalidOperationException("Invalid OHLCV scaler length.");

        worker = new Worker(ModelLoader.Load(modelAsset), backend);
    }

    // rawWindow is 60 rows x 55 columns, in metadata.ohlcvColumns order.
    // The returned array is the next raw OHLCV bar, 11 x 5.
    public float[] PredictNextBar(float[] rawWindow)
    {
        int expected = metadata.sequenceLength * metadata.featureSize;
        if (rawWindow == null || rawWindow.Length != expected)
            throw new ArgumentException($"Expected {expected} OHLCV values.", nameof(rawWindow));

        float[] normalized = new float[expected];
        for (int i = 0; i < expected; i++)
        {
            int field = i % metadata.featureSize;
            float scale = Mathf.Max(metadata.scaler.scale[field], 1e-12f);
            normalized[i] = (rawWindow[i] - metadata.scaler.mean[field]) / scale;
        }

        using (var input = new Tensor<float>(
            new TensorShape(1, metadata.sequenceLength, metadata.featureSize), normalized))
        {
            worker.Schedule(input);
            var outputTensor = worker.PeekOutput() as Tensor<float>;
            if (outputTensor == null)
                throw new InvalidOperationException("The model output is not Tensor<float>.");

            float[] rawBar = outputTensor.DownloadToArray();
            if (rawBar.Length != SectorCount * FieldCount)
                throw new InvalidOperationException("Expected output shape [1, 11, 5].");

            return rawBar;
        }
    }

    private void OnDestroy()
    {
        worker?.Dispose();
        worker = null;
    }
}
```

## 6. 複数日生成

生成したバーを次の入力行として末尾へ追加し、最古の行を削除して60行を維持します。

```text
過去60行の生OHLCV
    ↓ scalerで標準化
LSTM: [1, 60, 55] → [1, 11, 5]
    ↓ 相対値を前バーから生OHLCVへ再構成
次の11セクターOHLCV
    ↓ 入力ウィンドウ末尾へ追加
再び推論
```

標準化した外部入力を事前に±6へクリップしないでください。新モデルは内部で終値・直近出来高に対する相対値へ変換し、その後で制限します。上記C#例はノイズなしのモデル用です。

Pythonと同じ確率的生成には `lstm_model.stochastic.onnx` を使います。metadataの `residualBank [N,11,5]` から開始日を選び、市場全体の残差を5日続けて読み出します（末尾では先頭へ折り返す）。5日後に開始日を選び直します。全項目へ `generation.stochasticScale`、出来高へさらに `generation.volumeStochasticScale` を掛けて `residual` 入力へ渡します。セクターや項目ごとに別の日を選ぶと依存関係が失われます。同じ推論結果を比較する場合、seedの値だけでなく残差の抽出インデックスもPythonと揃えてください。ONNX Runtimeでの一致はテスト済みですが、Unity Editor実機でのロード・実行確認は別途必要です。

## 7. 保存形式

`data/generated/generated_prices.parquet`は、既存利用者向けのセクターclose列に加えて、次の55列を持ちます。

```text
energy                    # energy__closeの互換alias
energy__open
energy__high
energy__low
energy__close
energy__volume
```

全セクターのOHLCV専用ファイルは`data/generated/generated_ohlcv.parquet`です。

## 8. よくあるエラー

- `Expected OHLCV input shape [1, 60, 55]`: 旧215特徴量ウィンドウを渡しています。`feature_schema.json`の`ohlcv_columns`だけを使ってください。
- `Checkpoint ... is not an OHLCV-input model`: GRUまたは旧LSTMチェックポイントです。`train`を再実行してください。
- `Expected output shape [1, 11, 5]`: ONNXとmetadataが別バージョンです。同じexportで作ったファイルをコピーしてください。
- 値がNaN/Infになる: 入力55列の全値、scalerのmean/scale、出力の逆変換を確認してください。

## 9. 動作確認チェックリスト

- [ ] ONNX演算子に`LSTM`があり、`GRU`がない
- [ ] 入力が`[1, 60, 55]`
- [ ] 出力名が`ohlcv`
- [ ] 出力が`[1, 11, 5]`
- [ ] `ohlcvColumns`の列順を変更していない
- [ ] mean/scaleで入力を標準化している（出力はすでに生OHLCV）
- [ ] Open/Close/Volumeが正値
- [ ] High/LowがOpen/Closeを包含している
- [ ] Workerと入力Tensorを適切にDisposeしている

## 10. 参考ファイル

- [ONNX出力処理](../src/stock_sim/export_onnx.py)
- [LSTMモデル定義](../src/stock_sim/model.py)
- [OHLCV集約](../src/stock_sim/features.py)
- [Python側の生成処理](../src/stock_sim/generate.py)
- [OHLCV列順](../data/processed/feature_schema.json)
