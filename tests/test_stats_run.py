"""Integration tests for the stats orchestrator: FDR scope and the covariate block."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from symptommonster.stats.run import _assign_fdr, run_stats


def test_assign_fdr_global_is_more_conservative_than_within_drug():
    # Two groups, one pair each. Within-drug, each group is its own family of one,
    # so q == p; pooling both into one global family can only inflate the q-values.
    def make():
        return {"a": [{"p_value": 0.001}], "b": [{"p_value": 0.04}]}

    within = make()
    _assign_fdr(within, "p_value", "q_value", "significant", 0.10, "within-drug")
    glob = make()
    _assign_fdr(glob, "p_value", "q_value", "significant", 0.10, "global")

    assert within["a"][0]["q_value"] == pytest.approx(0.001)
    assert within["b"][0]["q_value"] == pytest.approx(0.04)
    assert glob["a"][0]["q_value"] >= within["a"][0]["q_value"]
    assert glob["b"][0]["q_value"] >= within["b"][0]["q_value"]
    within_total = within["a"][0]["q_value"] + within["b"][0]["q_value"]
    global_total = glob["a"][0]["q_value"] + glob["b"][0]["q_value"]
    assert global_total > within_total


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_run_stats_with_covariates_writes_residual_columns(tmp_path):
    n = 24
    signal = [{"patient_id": f"p{i}", "group": "a", "symptoms": ["fatigue"]} for i in range(n)]
    noise = [
        {"patient_id": f"p{i}", "group": "a", "symptoms": ["fatigue"] if i % 6 == 0 else []}
        for i in range(n)
    ]
    signal_path = tmp_path / "signal.jsonl"
    noise_path = tmp_path / "noise.jsonl"
    _write_jsonl(signal_path, signal)
    _write_jsonl(noise_path, noise)

    covariates = pd.DataFrame(
        {"patient_id": [f"p{i}" for i in range(n)], "site": ["A", "B"] * (n // 2)}
    )
    cov_path = tmp_path / "covariates.csv"
    covariates.to_csv(cov_path, index=False)

    out_dir = tmp_path / "out"
    run_stats(signal=str(signal_path), noise=str(noise_path), out_dir=str(out_dir), covariates=str(cov_path))

    table = pd.read_csv(out_dir / "stats" / "a.csv")
    assert {"lr_resid_mean", "lr_resid_p", "lr_resid_q", "lr_resid_significant"} <= set(table.columns)
    assert table["lr_resid_p"].between(0.0, 1.0).all()
    # Fatigue is present for every signal patient but a fraction of noise patients,
    # so the covariate-adjusted residual should be positive.
    fatigue = table[table["symptom"] == "fatigue"].iloc[0]
    assert fatigue["lr_resid_mean"] > 0


def test_run_stats_without_covariates_omits_residual_columns(tmp_path):
    signal = [{"patient_id": f"p{i}", "group": "a", "symptoms": ["fatigue"]} for i in range(10)]
    noise = [{"patient_id": f"p{i}", "group": "a", "symptoms": []} for i in range(10)]
    signal_path = tmp_path / "signal.jsonl"
    noise_path = tmp_path / "noise.jsonl"
    _write_jsonl(signal_path, signal)
    _write_jsonl(noise_path, noise)

    out_dir = tmp_path / "out"
    run_stats(signal=str(signal_path), noise=str(noise_path), out_dir=str(out_dir))

    table = pd.read_csv(out_dir / "stats" / "a.csv")
    assert "lr_resid_mean" not in table.columns
    assert "summary.csv" in {p.name for p in out_dir.iterdir()}
