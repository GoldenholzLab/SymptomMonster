"""Manuscript tables built from documented-format inputs.

Table 1 formats whatever demographic columns are provided into a tidy frame.
Table 2 ranks the top significant signals by signal-minus-noise. Both are pure
DataFrame transforms with no embedded values and no hard-coded column names
beyond the documented signal-summary fields.
"""

from __future__ import annotations

import pandas as pd


def build_table1(demographics: pd.DataFrame) -> pd.DataFrame:
    """Format the provided demographics table generically.

    The input has a `group` column plus arbitrary numeric/percentage columns.
    Float columns are rounded to one decimal for display; everything else passes
    through unchanged. No column is assumed by name beyond `group`, which (when
    present) is moved to the front.
    """
    if demographics is None or demographics.empty:
        return pd.DataFrame()

    table = demographics.copy()
    float_cols = table.select_dtypes(include="float").columns
    table[float_cols] = table[float_cols].round(1)

    if "group" in table.columns:
        ordered = ["group"] + [c for c in table.columns if c != "group"]
        table = table[ordered]
    return table.reset_index(drop=True)


def build_table2(summary: pd.DataFrame, *, top_n: int = 25) -> pd.DataFrame:
    """Return the top `top_n` significant signals by signal-minus-noise.

    `summary` follows the documented schema: `group`, `symptom`,
    `signal_minus_noise`, `significant` (optional `signal_pct`, `q_value`).
    Rows are filtered to significant findings when that column exists, then
    sorted by descending signal-minus-noise. Missing inputs yield an empty frame.
    """
    if summary is None or summary.empty or "signal_minus_noise" not in summary.columns:
        return pd.DataFrame()

    rows = summary.copy()
    if "significant" in rows.columns:
        rows = rows[rows["significant"].fillna(False).astype(bool)]

    rows = rows.sort_values("signal_minus_noise", ascending=False).head(top_n)

    columns = [
        c
        for c in ("group", "symptom", "signal_pct", "signal_minus_noise", "q_value")
        if c in rows.columns
    ]
    return rows[columns].reset_index(drop=True)
