"""Export the LSTM inference graph for Unity Sentis/Inference Engine."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path

import numpy as np

from .config import config_path, ensure_parent, load_config, setup_logging
from .constants import OHLCV_FIELDS, SECTOR_IDS
from .generate import _load_checkpoint, _relative_scaler_from_checkpoint
from .inference import OHLCVInference
from .ohlcv import RELATIVE_OHLCV_FIELDS
from .preprocessing import FeatureScaler

LOGGER = logging.getLogger(__name__)


def export_onnx_model(
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    output_path: str | Path,
    *,
    opset_version: int = 15,
    metadata_path: str | Path | None = None,
) -> Path:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for ONNX export") from exc
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the onnx extra before exporting") from exc
    scaler = FeatureScaler.load(scaler_path)
    device = torch.device("cpu")
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    sequence_length = int(checkpoint.get("sequence_length", 60))
    volume_lookback = int(checkpoint.get("volume_lookback", 20))
    if volume_lookback < 1 or volume_lookback > sequence_length:
        raise ValueError("The checkpoint volume_lookback must be between one and sequence_length")
    ohlcv_columns = list(checkpoint["ohlcv_columns"])
    expected_output_size = len(SECTOR_IDS) * len(OHLCV_FIELDS)
    if len(scaler.columns) != expected_output_size or len(ohlcv_columns) != expected_output_size:
        raise ValueError("The OHLCV scaler/checkpoint must contain 11 sectors x 5 fields")
    if list(scaler.columns) != ohlcv_columns:
        raise ValueError("The OHLCV scaler columns do not match the checkpoint order")
    relative_columns = list(checkpoint["relative_columns"])
    relative_metadata = checkpoint["relative_scaler_metadata"]
    relative_mean = np.asarray(relative_metadata["mean"], dtype=np.float32)
    relative_scale = np.asarray(relative_metadata["scale"], dtype=np.float32)
    if (
        len(relative_columns) != expected_output_size
        or relative_mean.shape != (expected_output_size,)
        or relative_scale.shape != (expected_output_size,)
    ):
        raise ValueError("The relative target scaler must contain 11 sectors x 5 fields")
    generation_config = checkpoint.get("generation_config", {})
    relative_clip = np.asarray(
        generation_config.get("relative_clip", [0.12, 0.12, 0.08, 0.08, 1.0]),
        dtype=np.float32,
    )
    if relative_clip.shape != (len(RELATIVE_OHLCV_FIELDS),) or (relative_clip <= 0).any():
        raise ValueError("relative_clip must contain five positive limits")

    export_model = OHLCVInference(model, scaler, _relative_scaler_from_checkpoint(checkpoint),
                                 volume_lookback, relative_clip,
                                 generation_config.get("feature_z_clip", 6.0),
                                 volume_anchor=checkpoint.get("volume_anchor"),
                                 volume_anchor_strength=generation_config.get("volume_anchor_strength", .1),
                                 volume_daily_limit=generation_config.get("volume_daily_limit", 3.)).eval()
    dummy = torch.zeros((1, sequence_length, len(scaler.columns)), dtype=torch.float32)
    output = ensure_parent(output_path)
    export_kwargs = {
        "input_names": ["features"],
        "output_names": ["ohlcv"],
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    # Unity consumes one market window at a time. Keep the exported LSTM
    # batch dimension fixed at one because the initial hidden/cell states are
    # internal constants rather than explicit ONNX inputs.
    # Newer PyTorch releases default to the torch.export-based exporter, which
    # may require an additional onnxscript package. The legacy exporter is
    # intentionally selected when the keyword exists because it produces the
    # ONNX LSTM operator supported by Unity Sentis/Inference Engine.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(export_model, dummy, output, **export_kwargs)
    import onnx

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    # Optional explicit noise input allows Unity to reproduce Python sampling.
    stochastic_output = output.with_name(output.stem + ".stochastic.onnx")
    stochastic_kwargs = dict(export_kwargs, input_names=["features", "residual"])
    torch.onnx.export(export_model, (dummy, torch.zeros(1, 11, 5)), stochastic_output, **stochastic_kwargs)
    onnx.checker.check_model(onnx.load(str(stochastic_output)))
    metadata_output = ensure_parent(metadata_path or output.with_suffix(".metadata.json"))
    scaler_mean = np.asarray(scaler.mean_, dtype=np.float32).tolist()
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float32).tolist()
    metadata = {
        "schemaVersion": 4,
        "modelType": "LSTM_RELATIVE_OHLCV",
        "inputType": "OHLCV",
        "outputType": "OHLCV",
        "onnx": {
            "inputName": "features",
            "outputName": "ohlcv",
            "inputShape": [1, sequence_length, len(scaler.columns)],
            "outputShape": [1, len(SECTOR_IDS), len(OHLCV_FIELDS)],
            "opset": opset_version,
        },
        "sequenceLength": sequence_length,
        "volumeLookback": volume_lookback,
        "volumeAnchor": checkpoint.get("volume_anchor"),
        "featureSize": len(scaler.columns),
        "featureColumns": list(scaler.columns),
        "ohlcvFields": list(OHLCV_FIELDS),
        "ohlcvColumns": ohlcv_columns,
        "relativeFields": list(RELATIVE_OHLCV_FIELDS),
        "relativeColumns": relative_columns,
        "sectorIds": list(SECTOR_IDS),
        "outputSemantics": {
            "input": "standardized raw OHLCV values in ohlcvColumns order",
            "head": "standardized gap/body/upper-wick/lower-wick/log-volume-ratio values",
            "output": "raw next OHLCV reconstructed using the previous close and anchored geometric-mean volume",
            "reconstruction": "open=previousClose*exp(gap); close=open*exp(body); high=max(open,close)*exp(upperWick); low=min(open,close)*exp(-lowerWick); volumeReference=exp((1-alpha)*mean(log(recentVolume))+alpha*log(volumeAnchor)); volume=volumeReference*exp(logVolumeRatio)",
            "postprocess": "relative clipping, non-negative wicks and daily volume limits are embedded in the graph; alpha=volumeAnchorStrength (zero for legacy checkpoints without anchors)",
        },
        "scaler": {"mean": scaler_mean, "scale": scaler_scale},
        "relativeTargetScaler": {"mean": relative_mean.tolist(), "scale": relative_scale.tolist()},
        "relativeNoiseScale": checkpoint.get("relative_noise_scale", []),
        "residualBank": checkpoint.get("residual_bank", []),
        "inputNormalization": "window-relative log OHLCV" if model.normalize_window else "global scaler",
        "stochasticOnnx": {"file": stochastic_output.name, "residualInput": "residual", "residualShape": [1, 11, 5]},
        "generation": {
            "seed": checkpoint.get("generation_seed"),
            "featureZClip": generation_config.get("feature_z_clip", 6.0),
            "stochasticScale": generation_config.get("stochastic_scale", 1.5),
            "volumeStochasticScale": generation_config.get("volume_stochastic_scale", 0.25),
            "relativeClip": relative_clip.tolist(),
            "residualBlockLength": generation_config.get("residual_block_length", 5),
            "volumeAnchorStrength": generation_config.get("volume_anchor_strength", .1),
            "volumeDailyLimit": generation_config.get("volume_daily_limit", 3.),
        },
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Exported Unity metadata: %s", metadata_output)
    LOGGER.info("Exported ONNX model: %s, input=%s, output=%s", output, graph.graph.input[0].type.tensor_type.shape, graph.graph.output[0].type.tensor_type.shape)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument("--metadata", default=None, help="Unity JSON metadata path")
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    output = args.output or str(config_path(config, "checkpoint").with_suffix(".onnx"))
    export_onnx_model(
        config_path(config, "checkpoint"),
        config_path(config, "scaler"),
        output,
        opset_version=args.opset,
        metadata_path=args.metadata,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
