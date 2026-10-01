from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(df: pd.DataFrame, clf: pd.DataFrame, reg: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        df.groupby("period", observed=True)
        .agg(disbursement=("digital_disbursement_crore", "sum"), loans=("digital_loan_count", "sum"))
        .reset_index()
    )
    summary["avg_ticket"] = summary["disbursement"] * 1e7 / summary["loans"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["period"].astype(str), summary["disbursement"], marker="o")
    ax.set_title("Observed Digital Disbursement by Period")
    ax.set_ylabel("₹ crore")
    ax.set_xlabel("Period")
    fig.tight_layout()
    fig.savefig(output_dir / "01_disbursement_trend.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["period"].astype(str), summary["loans"], marker="o")
    ax.set_title("Observed Digital Loan Count by Period")
    ax.set_ylabel("Loan count")
    ax.set_xlabel("Period")
    fig.tight_layout()
    fig.savefig(output_dir / "02_loan_count_trend.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["period"].astype(str), summary["avg_ticket"], marker="o")
    ax.set_title("Observed Weighted Average Ticket by Period")
    ax.set_ylabel("₹")
    ax.set_xlabel("Period")
    fig.tight_layout()
    fig.savefig(output_dir / "03_average_ticket_trend.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(clf["model"], clf["f1"])
    ax.set_title("Classification Model Comparison")
    ax.set_ylabel("F1")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "04_classification_model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(reg["model"], reg["RMSE_crore"])
    ax.set_title("Regression Model Comparison — RMSE")
    ax.set_ylabel("₹ crore")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(output_dir / "05_regression_rmse_comparison.png", dpi=150)
    plt.close(fig)
