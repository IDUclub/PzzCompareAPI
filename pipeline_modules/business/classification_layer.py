from __future__ import annotations

from typing import Any

import pandas as pd


def ensure_classification_columns(df: pd.DataFrame, context: Any=None) -> pd.DataFrame:
    """
    Ensure minimal result columns required by postprocessing are present.
    """
    result = df.copy()
    defaults: dict[str, Any] = {
        "PZZ_VRI_VERDICT": "unclear",
        "Статус": "Требуется ручная проверка",
        "PZZ_REASON": "",
        "MATCH_METHOD": "not_classified",
        "MATCHED_VRI_NAME": "",
        "MATCHED_VRI_CODE": "",
        "PZZ_NOT_ALLOWED_TOP5_CANDIDATES": "",
    }
    for col, value in defaults.items():
        if col not in result.columns:
            result[col] = value
    return result
