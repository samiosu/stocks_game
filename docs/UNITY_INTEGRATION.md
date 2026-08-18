# `.onnx` を Unity で使用するマニュアル

この文書では、`models/gru_model.onnx` を Unity の Sentis / Inference Engine で読み込み、日本株11セクターの値動きを確率的に生成する方法を説明します。

## 1. まず理解しておくこと

このONNXモデルは「次の値段」を直接出力しません。

```text
入力: 過去60営業日の特徴量
出力: 各セクターの mu と log_sigma
乱数: Unity側で生成
価格: Unity側で exp(log return) を使って更新
```

モデルの出力は次の意味です。

```text
mu        = 次の対数リターンの平均
log_sigma = 次の対数リターンの標準偏差を表すモデル出力
sigma     = softplus(log_sigma) + 1e-5
return    = mu + sigma * N(0, 1)
price     = previous_price * exp(return)
```

したがって、ONNXを読み込むだけでは株価チャートは生成されません。Unity側で、入力特徴量の作成、標準化、乱数サンプリング、価格更新を実装する必要があります。

## 2. 使用するファイル

Python側で学習・出力した次のファイルをUnityプロジェクトへコピーします。

| ファイル | 用途 |
| --- | --- |
| `models/gru_model.onnx` | Unityで実行するニューラルネットワーク |
| `models/gru_model.metadata.json` | scaler、特徴量順、factor loading、seed、生成式をまとめたUnity用メタデータ |
| `data/processed/feature_schema.json` | 215個の特徴量の順序 |
| `models/feature_scaler.pkl` | 学習データでfitした標準化パラメータ。Unity用JSONへ変換する |
| `data/processed/sector_data.parquet` | 初期60営業日の特徴量を作る元データ。UnityへはJSON等へ変換する |

`models/gru_model.pt` はPyTorch用チェックポイントであり、Unityには通常不要です。

推奨するUnity側の配置例は次のとおりです。

```text
Assets/
└── Models/
    ├── gru_model.onnx
    ├── feature_schema.json
    ├── gru_model.metadata.json
    └── initial_feature_window.json
```

## 3. Unityプロジェクトの準備

Unity 6では、Package Managerから `com.unity.ai.inference` を追加します。Unity公式ドキュメント上では、Unity 6000.0向けに Sentis 2.4.1 がリリースされています。

1. Unityで `Window > Package Manager` を開く
2. `+` ボタンから `Add package by name...` を選ぶ
3. Package nameに `com.unity.ai.inference` を入力する
4. 必要に応じてバージョン `2.4.1` を指定する
5. `models/gru_model.onnx` をUnityプロジェクトの `Assets/Models/` へコピーする

UnityがONNXをインポートすると、Projectウィンドウ上でモデルアセットとして扱えるようになります。外部weightファイルがあるモデルではONNXと同じフォルダーへ置きますが、このプロジェクトのモデルは通常単一ファイルです。

公式資料:

- [Sentis概要](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.4/manual/index.html)
- [ONNXファイルのインポート](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.4/manual/import-a-model-file.html)
- [Workerの作成](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.4/manual/create-an-engine.html)

### ONNX opsetについて

現在の `models/gru_model.onnx` はopset 17で出力されています。Sentisのバージョンによって対応opsetが異なるため、インポートエラーが出る場合はSentis互換性を優先してopset 15で再出力してください。

```bash
python -m stock_sim.export_onnx \
  --opset 15 \
  --output models/gru_model_sentis.onnx
```

Unityへコピーするのは、再出力した `gru_model_sentis.onnx` です。

## 4. Unity用メタデータを作成する

`feature_scaler.pkl` はPythonのpickle形式なので、Unityで直接読むことはできません。ONNX出力コマンドを実行すると、同じ場所にUnityで読めるメタデータJSONも自動生成されます。

```bash
python -m stock_sim.export_onnx \
  --config config/config.yaml \
  --output models/gru_model.onnx \
  --metadata models/gru_model.metadata.json
```

このJSONには `mean` と `scale`、`featureColumns`、`factorLoadings`、`commonNoiseWeight`、生成seed、`softplus`を使うsigma式が含まれます。別の平均・標準偏差をUnity側で作らないでください。`factorLoadings` はセクター-majorの一次元配列で、位置は `sector * factorCount + factor` です。

## 5. 初期入力ウィンドウを作る

最初の推論には、時系列順の過去60行が必要です。UnityでParquetを読む処理を用意しない場合は、次のコマンドでJSONへ変換できます。

```bash
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

schema = json.loads(Path("data/processed/feature_schema.json").read_text(encoding="utf-8"))
columns = schema["feature_columns"]
frame = pd.read_parquet("data/processed/sector_data.parquet").sort_values("date").tail(60)
if len(frame) != 60:
    raise RuntimeError(f"60 rows are required, got {len(frame)}")

payload = {
    "sequenceLength": 60,
    "featureSize": len(columns),
    "features": frame[columns].to_numpy(dtype="float32").reshape(-1).tolist(),
}
Path("data/processed/initial_feature_window.json").write_text(
    json.dumps(payload),
    encoding="utf-8",
)
print("wrote data/processed/initial_feature_window.json")
PY
```

ここで出力される `features` は標準化前の値です。後述のC#コードが `mean` と `scale` を使って標準化します。

## 6. モデルの入出力仕様

現在のモデルの実測仕様は次のとおりです。

| 項目 | 値 |
| --- | --- |
| 入力名 | `features` |
| 入力型 | `float32` |
| 入力形状 | `[batch, 60, 215]` |
| Unityでの通常のbatch | `1` |
| 出力名 | `params` |
| 出力型 | `float32` |
| 出力形状 | `[batch, 11, 2]` |

入力配列は、時刻を古い順に並べたrow-major配列です。Unityの1次元配列では次の位置になります。

```text
features[t * 215 + f]

t = 0..59  : 時系列。0が最も古い行、59が最新行
f = 0..214 : feature_schema.jsonのfeature_columnsの順番
```

出力は次の位置に格納されます。

```text
params[sectorIndex * 2 + 0] = mu
params[sectorIndex * 2 + 1] = log_sigma
```

### セクター順

この順番は固定です。表示名ではなく英語IDを使います。

```text
0  energy
1  materials
2  industrials
3  consumer_discretionary
4  consumer_staples
5  health_care
6  financials
7  information_technology
8  communication_services
9  utilities
10 real_estate
```

### 215特徴量の内訳

`data/processed/feature_schema.json` の `feature_columns` が唯一の正解です。順番を手入力で並べ替えないでください。

```text
11セクター × 9個のセクター特徴量 = 99
市場共通特徴量                  = 8
9イベント × (全体1 + セクター11) = 108
合計                            = 215
```

各セクターの9特徴量の順序は次のとおりです。

```text
return_1d
return_5d
oc_change
hl_range
volume_log_change
vol_5d
vol_20d
excess_topix
excess_n225
```

市場共通特徴量は次の8個です。

```text
market__n225_return_1d
market__topix_return_1d
market__return_spread
market__n225_volatility
market__topix_volatility
market__up_sector_count
market__down_sector_count
market__sector_cross_std
```

イベント特徴量は、各イベントについて全体用1列とセクター別11列を持ちます。イベントを使用しない場合は、108列すべてを `0` にしてください。

## 7. Unity用C#実装例

次のスクリプトを `Assets/Scripts/StockMarketOnnxRunner.cs` として保存します。これは、標準化前の60行×215列を受け取り、11セクターの次の価格を一日分生成する最小実装です。

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
        public string[] featureColumns;
        public float[] mean;
        public float[] scale;
        public float[] returnLow;
        public float[] returnHigh;
        public string[] sectorIds;
        public int factorCount;
        public float[] factorLoadings;
        public float commonNoiseWeight;
        public Generation generation;

        [Serializable]
        public sealed class Generation
        {
            public int seed;
            public bool hardClip;
            public float returnSoftClip;
            public float volatilityPersistence;
            public float volatilityShockScale;
        }
    }

    [Header("Assets")]
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private TextAsset metadataJson;

    [Header("Runtime")]
    [SerializeField] private BackendType backend = BackendType.CPU;
    [SerializeField] private int randomSeed = 42;

    private const int SectorCount = 11;
    private Worker worker;
    private Metadata metadata;
    private System.Random random;
    private float volatilityState;

    private void Awake()
    {
        if (modelAsset == null || metadataJson == null)
            throw new InvalidOperationException("ModelAsset and metadataJson are required.");

        metadata = JsonUtility.FromJson<Metadata>(metadataJson.text);
        ValidateMetadata();

        Model runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, backend);
        random = new System.Random(randomSeed);
    }

    private void ValidateMetadata()
    {
        if (metadata == null || metadata.sequenceLength != 60 || metadata.featureSize != 215)
            throw new InvalidOperationException("Expected sequenceLength=60 and featureSize=215.");
        if (metadata.mean == null || metadata.mean.Length != metadata.featureSize)
            throw new InvalidOperationException("Invalid scaler mean length.");
        if (metadata.scale == null || metadata.scale.Length != metadata.featureSize)
            throw new InvalidOperationException("Invalid scaler scale length.");
        if (metadata.returnLow == null || metadata.returnLow.Length != SectorCount)
            throw new InvalidOperationException("Invalid returnLow length.");
        if (metadata.returnHigh == null || metadata.returnHigh.Length != SectorCount)
            throw new InvalidOperationException("Invalid returnHigh length.");
        if (metadata.factorCount != 3 || metadata.factorLoadings == null || metadata.factorLoadings.Length != SectorCount * metadata.factorCount)
            throw new InvalidOperationException("Invalid factor loading metadata.");
        if (metadata.generation == null)
            metadata.generation = new Metadata.Generation
            {
                returnSoftClip = 0.08f,
                volatilityPersistence = 0.9f,
                volatilityShockScale = 0.18f,
            };
        if (metadata.commonNoiseWeight < 0.0f || metadata.commonNoiseWeight > 1.0f)
            throw new InvalidOperationException("Invalid commonNoiseWeight.");
    }

    // rawWindow is [60, 215], flattened in row-major order.
    public float[] PredictParameters(float[] rawWindow)
    {
        int expected = metadata.sequenceLength * metadata.featureSize;
        if (rawWindow == null || rawWindow.Length != expected)
            throw new ArgumentException($"Expected {expected} raw feature values.", nameof(rawWindow));

        float[] normalized = Normalize(rawWindow);
        var input = new Tensor<float>(
            new TensorShape(1, metadata.sequenceLength, metadata.featureSize),
            normalized);

        try
        {
            worker.Schedule(input);
            var outputTensor = worker.PeekOutput() as Tensor<float>;
            if (outputTensor == null)
                throw new InvalidOperationException("The model output is not Tensor<float>.");

            // This is a blocking readback. Use ReadbackAndCloneAsync in a
            // frame-sensitive production path; see section 11.
            float[] values = outputTensor.DownloadToArray();
            if (values.Length != SectorCount * 2)
                throw new InvalidOperationException("Expected output shape [1, 11, 2].");
            return values;
        }
        finally
        {
            input.Dispose();
        }
    }

    // currentPrices and the returned array are in the fixed sector order.
    public float[] GenerateNextPrices(
        float[] rawWindow,
        float[] currentPrices,
        float volatilityScale = 1.0f)
    {
        if (currentPrices == null || currentPrices.Length != SectorCount)
            throw new ArgumentException("Expected 11 current prices.", nameof(currentPrices));
        if (volatilityScale < 0.0f)
            throw new ArgumentOutOfRangeException(nameof(volatilityScale));

        float[] parameters = PredictParameters(rawWindow);
        float[] nextPrices = new float[SectorCount];
        float[] factorNoise = FactorNoise();
        float volatilityMultiplier = VolatilityMultiplier();
        for (int sector = 0; sector < SectorCount; sector++)
        {
            float mu = parameters[sector * 2];
            float logSigma = parameters[sector * 2 + 1];
            float sigma = Softplus(logSigma) + 1e-5f;
            float idiosyncratic = NextStandardNormal();
            float common = 0.0f;
            for (int factor = 0; factor < metadata.factorCount; factor++)
                common += factorNoise[factor] * metadata.factorLoadings[sector * metadata.factorCount + factor];
            float weight = metadata.commonNoiseWeight;
            float noise = weight * common + Mathf.Sqrt(Mathf.Max(0.0f, 1.0f - weight * weight)) * idiosyncratic;
            float logReturn = mu + sigma * volatilityScale * volatilityMultiplier * noise;

            if (metadata.generation.returnSoftClip > 0.0f)
            {
                float softClip = metadata.generation.returnSoftClip;
                logReturn = softClip * Mathf.Tanh(logReturn / softClip);
            }
            if (metadata.generation.hardClip)
                logReturn = Mathf.Clamp(logReturn, metadata.returnLow[sector], metadata.returnHigh[sector]);
            nextPrices[sector] = currentPrices[sector] * Mathf.Exp(logReturn);
        }
        return nextPrices;
    }

    private float[] FactorNoise()
    {
        float[] values = new float[metadata.factorCount];
        for (int i = 0; i < values.Length; i++)
            values[i] = NextStandardNormal();
        return values;
    }

    private float VolatilityMultiplier()
    {
        float persistence = Mathf.Clamp(metadata.generation.volatilityPersistence, 0.0f, 0.999f);
        volatilityState = persistence * volatilityState
            + Mathf.Sqrt(1.0f - persistence * persistence) * NextStandardNormal();
        float shock = Mathf.Max(0.0f, metadata.generation.volatilityShockScale);
        return Mathf.Exp(shock * volatilityState - 0.5f * shock * shock);
    }

    private static float Softplus(float value)
    {
        if (value > 20.0f)
            return value;
        return Mathf.Log(1.0f + Mathf.Exp(value));
    }

    private float[] Normalize(float[] rawWindow)
    {
        float[] result = new float[rawWindow.Length];
        for (int i = 0; i < rawWindow.Length; i++)
        {
            int feature = i % metadata.featureSize;
            float scale = metadata.scale[feature];
            if (scale < 1e-12f)
                scale = 1.0f;
            result[i] = (rawWindow[i] - metadata.mean[feature]) / scale;
        }
        return result;
    }

    private float NextStandardNormal()
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    private void OnDestroy()
    {
        worker?.Dispose();
        worker = null;
    }
}
```

### スクリプトの設定

1. 空のGameObjectを作成する
2. `StockMarketOnnxRunner`をアタッチする
3. `Model Asset`へProjectウィンドウのONNXアセットを割り当てる
4. `Metadata Json`へ `gru_model.metadata.json` を割り当てる
5. 最初は `BackendType.CPU` で動作確認する
6. 対応プラットフォームで性能を確認した後、`GPUCompute`を試す

Sentisでは、モデルをロードして `Worker` を作り、`Schedule` で推論を実行します。`PeekOutput`の戻り値はWorkerが所有するため、上のコードでは出力Tensorを個別にDisposeしていません。一方、入力Tensorはこのコードで作成したものなので、読み出し後にDisposeしています。

## 8. JSONから初期ウィンドウを読む

`initial_feature_window.json` を `TextAsset` として読み込む場合の例です。

```csharp
using System;
using UnityEngine;

[Serializable]
public sealed class InitialFeatureWindow
{
    public int sequenceLength;
    public int featureSize;
    public float[] features;
}

// 例:
// var window = JsonUtility.FromJson<InitialFeatureWindow>(asset.text);
// var next = runner.GenerateNextPrices(window.features, currentPrices);
```

`features`の長さは必ず `60 * 215 = 12900` でなければなりません。

## 9. 複数日を生成する場合

一日分の価格を生成した後、同じ入力ウィンドウを毎回使い続けてはいけません。次の日の入力を作り、古い行を削除して新しい行を末尾へ追加します。

```text
最初の60行
    ↓ 推論
次の11セクターの対数リターンをサンプリング
    ↓
共通factor noiseとセクター固有noiseをloadingsで合成
    ↓
価格を price *= exp(return) で更新
    ↓
生成したリターンから次の特徴量行を作る
    ↓
先頭行を削除し、新しい行を末尾へ追加
    ↓
再び60行をONNXへ入力
```

次の値は少なくとも更新してください。

- 各セクターの `return_1d`
- 各セクターの直近5日 `return_5d`
- 各セクターの直近5日・20日 `vol_5d` / `vol_20d`
- `market__up_sector_count`
- `market__down_sector_count`
- `market__sector_cross_std`
- イベント特徴量

既存Python実装と同じ結果を目指す場合は、[features.py](../src/stock_sim/features.py) と [generate.py](../src/stock_sim/generate.py) の計算順・`ddof=0`・イベント減衰処理をUnity側へ移植してください。特に `return_5d` は直近5日リターンの合計、ボラティリティは母標準偏差です。

実データのOHLCV、出来高、TOPIX、日経平均までUnityで再現する必要がないゲーム構成では、生成開始後の未観測特徴量をゲーム用のルールで更新しても構いません。ただし、その場合はPythonで評価した分布と完全には一致しません。

## 10. イベント入力

イベントなしの場合は、全 `event__*` を0にします。

イベントを使う場合、基本的には次の値を設定します。

```text
event__<event_type>
event__<event_type>__<sector_id>
```

例えば、情報技術セクターへのAI投資イベントなら、次の列を設定します。

```text
event__ai_investment = intensity
event__ai_investment__information_technology = intensity
```

Python実装と同じ指数減衰にする場合は、イベント開始からの経過日数を `elapsed` として次を使います。

```text
amount = intensity * exp(-decay_rate * elapsed)
```

イベントの対象セクターが指定されない場合、Python実装では全11セクターを対象にします。列名と順番は必ず `feature_schema.json` の `event_columns` に合わせてください。

## 11. 性能とメモリ

- `Worker` は日ごとに作らず、ゲーム開始時に1回作って再利用する
- `Worker` はゲーム終了時に `Dispose` する
- GPU出力を毎フレーム `DownloadToArray` するとCPUへの同期が発生する
- 画面を止めたくない場合は `ReadbackAndCloneAsync` を使う
- 最初の `Schedule` はシェーダー・内部メモリの準備で遅くなる場合があるため、ロード時にダミー入力で一度実行する
- このモデルは通常batch 1で使用する

GPUでの出力をCPUへ読み出す非同期処理の例は、[Sentis公式の非同期readback資料](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.4/manual/read-output-async.html)を参照してください。

## 12. よくあるエラー

### `Expected [1, 60, 215]` などのshapeエラー

次を確認します。

- 入力が `float32` か
- 行数が60か
- 1行あたりの特徴量が215個か
- 行の順番が古い日から新しい日か
- `feature_schema.json` と同じ列順か

### ONNXのインポートに失敗する

opsetの不一致が考えられます。次のようにopset 15で再出力して再インポートします。

```bash
python -m stock_sim.export_onnx --opset 15 --output models/gru_model_sentis.onnx
```

### 出力が毎回同じになる

ONNXは同じ入力に対して同じ `mu` / `log_sigma` を返します。確率的な値動きにするには、Unity側で毎回異なる正規乱数 `epsilon` を生成してください。再現性が必要な場合は乱数seedを固定します。

### 価格が急激に0または極端な値になる

- `sigma = softplus(log_sigma) + 1e-5` を使っているか
- `return` を価格として直接加算していないか
- `price *= exp(return)` になっているか
- 通常は連続soft clipだけを使い、`hardClip`を不用意に有効化していないか
- `volatilityScale` を大きくしすぎていないか

### 実行時にNaNやInfが出る

入力215列のどこかにNaN、Inf、未初期化値がないか確認します。イベントなしの場合も、イベント列を欠落させず0で埋めてください。

## 13. 動作確認チェックリスト

- [ ] Unityに `com.unity.ai.inference` を追加した
- [ ] ONNXを `Assets/Models/` にコピーした
- [ ] `gru_model.metadata.json` をONNXと一緒にコピーした
- [ ] `feature_schema.json` の列順を変更していない
- [ ] 入力が `[1, 60, 215]` になっている
- [ ] 出力が `[1, 11, 2]` になっている
- [ ] `params[..., 0]` を `mu` として扱っている
- [ ] `params[..., 1]` を `log_sigma` として扱っている
- [ ] `softplus(log_sigma) + 1e-5` でsigmaを計算している
- [ ] `factorLoadings` と `commonNoiseWeight` で共通乱数を混ぜている
- [ ] Unity側で正規乱数をサンプリングしている
- [ ] 価格更新に `exp(return)` を使っている
- [ ] WorkerとTensorを適切にDisposeしている

## 14. 参考ファイル

- [ONNX出力処理](../src/stock_sim/export_onnx.py)
- [GRUモデル定義](../src/stock_sim/model.py)
- [特徴量生成](../src/stock_sim/features.py)
- [Python側の生成処理](../src/stock_sim/generate.py)
- [特徴量の順序](../data/processed/feature_schema.json)
