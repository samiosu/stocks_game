"""Export the GRU inference graph for Unity Sentis/Inference Engine."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path

import numpy as np

from .config import config_path, ensure_parent, load_config, setup_logging
from .constants import SECTOR_IDS
from .generate import _load_checkpoint
from .preprocessing import FeatureScaler

LOGGER = logging.getLogger(__name__)


def export_onnx_model(
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    output_path: str | Path,
    *,
    opset_version: int = 17,
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
    dummy = torch.zeros((1, sequence_length, len(scaler.columns)), dtype=torch.float32)
    output = ensure_parent(output_path)
    model.eval()
    export_kwargs = {
        "input_names": ["features"],
        "output_names": ["params"],
        "dynamic_axes": {"features": {0: "batch"}, "params": {0: "batch"}},
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    # Newer PyTorch releases default to the torch.export-based exporter, which
    # may require an additional onnxscript package. The legacy exporter is
    # intentionally selected when the keyword exists because it is the most
    # widely supported path for GRU graphs consumed by Unity.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, dummy, output, **export_kwargs)
    import onnx

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    metadata_output = ensure_parent(metadata_path or output.with_suffix(".metadata.json"))
    scaler_mean = np.asarray(scaler.mean_, dtype=np.float32).tolist()
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float32).tolist()
    bounds = checkpoint.get("return_bounds", [])
    factor_metadata = checkpoint.get("factor_metadata", {})
    factor_loadings = factor_metadata.get("factor_loadings", [])
    factor_flat = [value for row in factor_loadings for value in row]
    generation_config = checkpoint.get("generation_config", {})
    metadata = {
        "schemaVersion": 2,
        "onnx": {
            "inputName": "features",
            "outputName": "params",
            "inputShape": [1, sequence_length, len(scaler.columns)],
            "outputShape": [1, len(SECTOR_IDS), 2],
            "opset": opset_version,
        },
        "sequenceLength": sequence_length,
        "featureSize": len(scaler.columns),
        "featureColumns": list(scaler.columns),
        "sectorIds": list(SECTOR_IDS),
        "returnUnit": "log_return",
        "outputSemantics": {
            "mu": "conditional mean log return",
            "logSigma": "bounded log standard deviation of log return",
            "sigmaFormula": "softplus(logSigma) + 1e-5",
            "samplingFormula": "mu + sigma * epsilon, epsilon ~ N(0, 1)",
            "priceFormula": "previousPrice * exp(logReturn)",
        },
        "scaler": {"mean": scaler_mean, "scale": scaler_scale},
        "returnBounds": bounds,
        "returnLow": [row[0] for row in bounds],
        "returnHigh": [row[1] for row in bounds],
        "factorNames": factor_metadata.get("factor_names", []),
        "factorCount": len(factor_metadata.get("factor_names", [])),
        "factorLoadings": factor_flat,
        "commonNoiseWeight": factor_metadata.get("common_noise_weight", 0.0),
        "factorExplainedVarianceRatio": factor_metadata.get("explained_variance_ratio", []),
        "generation": {
            "seed": checkpoint.get("generation_seed"),
            "hardClip": bool(generation_config.get("hard_clip", False)),
            "returnSoftClip": generation_config.get("return_soft_clip", 0.08),
            "featureZClip": generation_config.get("feature_z_clip", 6.0),
            "volatilityPersistence": generation_config.get("volatility_persistence", 0.9),
            "volatilityShockScale": generation_config.get("volatility_shock_scale", 0.18),
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
    parser.add_argument("--opset", type=int, default=17)
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
