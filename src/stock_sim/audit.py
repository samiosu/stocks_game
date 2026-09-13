"""Reproducible held-out OHLCV and long-generation diagnostics (no tuning)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import SECTOR_IDS
from .evaluate import heldout_starts
from .generate import _load_checkpoint, generate_price_paths
from .preprocessing import FeatureScaler


def metrics(bars, previous):
    closes = np.concatenate([previous[None, :, 3], bars[:, :, 3]], axis=0)
    returns = np.diff(np.log(closes), axis=0)
    gap = np.log(bars[:, :, 0] / closes[:-1])
    corr = np.corrcoef(returns.T)
    volume = bars[..., 4]
    return {
        "return_std": float(returns.std()),
        "sector_correlation": float(corr[~np.eye(11, dtype=bool)].mean()),
        "gap_abs_median": float(np.median(abs(gap))),
        "gap_abs_q95": float(np.quantile(abs(gap), .95)),
        "std_second_over_first_half": float(returns[len(returns)//2:].std() / returns[:len(returns)//2].std()),
        "max_sector_volume_range": float(np.max(volume.max(axis=0) / volume.min(axis=0))),
        "max_terminal_volume_ratio": float(np.max(volume[-1] / previous[:, 4])),
        "min_terminal_volume_ratio": float(np.min(volume[-1] / previous[:, 4])),
        "valid_ohlcv": bool(np.isfinite(bars).all() and (bars > 0).all()
                            and (bars[..., 1] >= np.maximum(bars[..., 0], bars[..., 3])).all()
                            and (bars[..., 2] <= np.minimum(bars[..., 0], bars[..., 3])).all()),
    }


def audit(output):
    torch.set_num_threads(1)
    model, checkpoint = _load_checkpoint('models/lstm_model.pt', torch.device('cpu'))
    scaler = FeatureScaler.load('models/feature_scaler.pkl')
    frame = pd.read_parquet('data/processed/sector_data.parquet').sort_values('date').reset_index(drop=True)
    columns = checkpoint['ohlcv_columns']
    bars = frame[columns].to_numpy(float).reshape(-1, 11, 5)
    valid = heldout_starts(frame, checkpoint['sequence_length'], 120, checkpoint['validation_end'])
    starts = valid[np.linspace(0, len(valid)-1, 3, dtype=int)]
    rows = []
    for start in starts:
        actual = metrics(bars[start:start+120], bars[start-1])
        for seed in (42, 123, 2026):
            generated = generate_price_paths(model, frame.iloc[:start], scaler, checkpoint, days=120, seed=seed)
            simulated = metrics(generated[columns].to_numpy().reshape(-1,11,5), bars[start-1])
            rows.append(dict(start_date=str(frame.iloc[start]['date']), seed=seed, real=actual, generated=simulated))
    stress = []
    for seed in (42, 123, 2026):
        generated = generate_price_paths(model, frame, scaler, checkpoint, days=1000, seed=seed)
        stress.append(dict(seed=seed, **metrics(generated[columns].to_numpy().reshape(-1,11,5), bars[-1])))
    summary = {kind: {key: float(np.mean([row[kind][key] for row in rows]))
                      for key in rows[0][kind] if key != 'valid_ohlcv'} for kind in ('real', 'generated')}
    result = dict(summary=summary, scenarios=rows, stress_1000_days=stress)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(summary=summary, stress_1000_days=stress), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
