"""Reproduce the descriptive analysis and one exploratory hypothesis test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart_failure_clinical_records_dataset.csv"
OUTPUT = ROOT / "output" / "summary.json"
CHART = ROOT / "output" / "ejection_fraction.svg"
SEED = 20260928
PERMUTATIONS = 50_000
BOOTSTRAPS = 20_000

EXPECTED_COLUMNS = {
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time",
    "DEATH_EVENT",
}


def load_data(path: Path = DATA) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if set(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected columns in source data")
    if frame.isna().any().any():
        raise ValueError("Missing values require a separate handling plan")
    if frame.duplicated().any():
        raise ValueError("Duplicate rows found")
    if not set(frame["DEATH_EVENT"].unique()) <= {0, 1}:
        raise ValueError("DEATH_EVENT must be binary")
    if not frame["ejection_fraction"].between(0, 100).all():
        raise ValueError("Ejection fraction must be a percentage")
    if not frame["time"].gt(0).all():
        raise ValueError("Follow-up time must be positive")
    return frame


def permutation_test(a: np.ndarray, b: np.ndarray, repeats: int, seed: int) -> float:
    """Two-sided permutation p-value for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    observed = abs(float(a.mean() - b.mean()))
    combined = np.concatenate((a, b))
    extreme = 0
    for _ in range(repeats):
        shuffled = rng.permutation(combined)
        difference = abs(float(shuffled[: len(a)].mean() - shuffled[len(a) :].mean()))
        extreme += difference >= observed - 1e-12
    return (extreme + 1) / (repeats + 1)


def bootstrap_interval(
    a: np.ndarray, b: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    """Percentile interval for mean(a) - mean(b), resampling within groups."""
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats)
    for index in range(repeats):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        differences[index] = sample_a.mean() - sample_b.mean()
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(lower), float(upper)


def write_chart(alive_mean: float, deceased_mean: float, alive_count: int, deceased_count: int) -> None:
    """Small dependency-free chart for the README."""
    scale = 9
    bars = [
        ("Без события", alive_mean, alive_count, 70, "#257b96"),
        ("С событием", deceased_mean, deceased_count, 140, "#c76a42"),
    ]
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="240" viewBox="0 0 760 240" role="img" aria-labelledby="title desc">',
        '<title id="title">Средняя фракция выброса по исходу</title>',
        '<desc id="desc">Описательное сравнение двух групп. Это не оценка эффекта лечения.</desc>',
        '<rect width="760" height="240" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="Arial,sans-serif" font-size="19" font-weight="bold" fill="#162e3b">Фракция выброса и исход наблюдения</text>',
    ]
    for label, value, count, y, color in bars:
        elements.extend([
            f'<text x="20" y="{y}" font-family="Arial,sans-serif" font-size="15" fill="#162e3b">{label} (n={count})</text>',
            f'<rect x="190" y="{y - 19}" width="{value * scale:.1f}" height="26" rx="4" fill="{color}"/>',
            f'<text x="{205 + value * scale:.1f}" y="{y}" font-family="Arial,sans-serif" font-size="15" fill="#162e3b">{value:.1f}%</text>',
        ])
    elements.extend([
        '<line x1="190" y1="182" x2="640" y2="182" stroke="#59707c"/>',
        '<text x="190" y="202" font-family="Arial,sans-serif" font-size="12" fill="#59707c">0%</text>',
        '<text x="622" y="202" font-family="Arial,sans-serif" font-size="12" fill="#59707c">50%</text>',
        '<text x="20" y="228" font-family="Arial,sans-serif" font-size="12" fill="#59707c">Группы не назначались случайно; сроки наблюдения различаются.</text>',
        '</svg>',
    ])
    CHART.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    frame = load_data()
    alive = frame.loc[frame["DEATH_EVENT"] == 0, "ejection_fraction"].to_numpy()
    deceased = frame.loc[frame["DEATH_EVENT"] == 1, "ejection_fraction"].to_numpy()

    result = {
        "source_rows": len(frame),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "outcome_counts": {str(key): int(value) for key, value in frame["DEATH_EVENT"].value_counts().sort_index().items()},
        "hypothesis": "Mean ejection fraction is equal across DEATH_EVENT groups",
        "statistic": "mean(deceased) - mean(alive), percentage points",
        "alive_mean": float(alive.mean()),
        "deceased_mean": float(deceased.mean()),
        "difference": float(deceased.mean() - alive.mean()),
        "permutation_p_two_sided": permutation_test(deceased, alive, PERMUTATIONS, SEED),
        "bootstrap_95_interval": list(bootstrap_interval(deceased, alive, BOOTSTRAPS, SEED + 1)),
        "permutations": PERMUTATIONS,
        "bootstraps": BOOTSTRAPS,
        "seed": SEED,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_chart(float(alive.mean()), float(deceased.mean()), len(alive), len(deceased))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
