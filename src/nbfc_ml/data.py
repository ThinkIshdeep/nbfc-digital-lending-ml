from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import CORE_DATASET, PERIODS, TRANSITION_DATASET

REQUIRED_COLUMNS = [
    "nbfc_respondent_id",
    "period",
    "digital_disbursement_crore",
    "digital_loan_count",
    "average_ticket_rupees",
]


def load_core_data(path: Path = CORE_DATASET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Core dataset not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["period"] = pd.Categorical(df["period"], categories=PERIODS, ordered=True)
    numeric = [
        "digital_disbursement_crore",
        "digital_loan_count",
        "average_ticket_rupees",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Defensive de-duplication: the source itself is already unique.
    df = (
        df.drop_duplicates(["nbfc_respondent_id", "period"])
        .sort_values(["nbfc_respondent_id", "period"])
        .reset_index(drop=True)
    )

    # Recalculate average ticket only where amount and count are available.
    mask = df["digital_disbursement_crore"].notna() & (df["digital_loan_count"] > 0)
    df.loc[mask, "average_ticket_rupees"] = (
        df.loc[mask, "digital_disbursement_crore"] * 1e7 / df.loc[mask, "digital_loan_count"]
    )
    return df


def validate_core_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Core dataset is empty")
    if df["nbfc_respondent_id"].isna().any():
        raise ValueError("NBFC respondent ID contains missing values")
    if df[["digital_disbursement_crore", "digital_loan_count", "average_ticket_rupees"]].lt(0).any().any():
        raise ValueError("Negative monetary/count values are not allowed")
    dupes = df.duplicated(["nbfc_respondent_id", "period"]).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate respondent-period rows")


def build_transition_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for nbfc, group in df.groupby("nbfc_respondent_id", sort=True):
        group = group.set_index("period").reindex(PERIODS)
        for i in range(len(PERIODS) - 1):
            current = group.loc[PERIODS[i]]
            nxt = group.loc[PERIODS[i + 1]]
            vals = [
                current["digital_disbursement_crore"],
                current["digital_loan_count"],
                current["average_ticket_rupees"],
                nxt["digital_disbursement_crore"],
            ]
            if any(pd.isna(v) for v in vals):
                continue

            amount = float(current["digital_disbursement_crore"])
            count = float(current["digital_loan_count"])
            ticket = float(current["average_ticket_rupees"])
            next_amount = float(nxt["digital_disbursement_crore"])
            rows.append(
                {
                    "nbfc_respondent_id": nbfc,
                    "from_period": PERIODS[i],
                    "to_period": PERIODS[i + 1],
                    "period_index": i,
                    "current_disbursement_crore": amount,
                    "current_loan_count": count,
                    "current_average_ticket_rupees": ticket,
                    "current_log_disbursement": np.log1p(amount),
                    "current_log_loan_count": np.log1p(count),
                    "current_log_ticket": np.log1p(ticket),
                    "next_disbursement_crore": next_amount,
                    "next_disbursement_increased": int(next_amount > amount),
                }
            )
    result = pd.DataFrame(rows)
    if TRANSITION_DATASET.parent.exists():
        result.to_csv(TRANSITION_DATASET, index=False)
    return result
