# Data Dictionary

## Core dataset: `data/processed/digital_lending_long_clean.csv`

| Field | Type | Meaning |
|---|---|---|
| `nbfc_respondent_id` | string | Anonymous respondent identifier retained from the cleaned RBI table |
| `period` | category/string | FY2017, FY2018, FY2019, FY2020, or 31 Dec 2020 observation |
| `digital_disbursement_crore` | float | Reported digital loan disbursement amount in ₹ crore |
| `digital_loan_count` | float | Reported number of digital loans |
| `average_ticket_rupees` | float | Derived average ticket size = amount × 1 crore / number of loans |

## ML transition dataset

A row represents a **current-to-next observed period transition for the same NBFC respondent**.

| Field | Meaning |
|---|---|
| `current_disbursement_crore` | current-period reported disbursement |
| `current_loan_count` | current-period reported loan count |
| `current_average_ticket_rupees` | current-period average ticket |
| `period_index` | 0→3 encoding of the current period |
| `next_disbursement_crore` | target for regression |
| `next_disbursement_increased` | target for classification: 1 if next amount > current amount, else 0 |

The `current_log_*` fields are retained as reproducibility/reference features but are not required by the baseline classification/regression model.

## Supporting datasets

- `rbi_nbfc_sectoral_credit_long.csv`: NBFC **outstanding credit** by sector and reference date. This is a stock variable, not a transaction-level disbursement flow.
- `rbi_dpi_clean.csv`: national-level RBI Digital Payments Index series; contextual digitalisation indicator, not a firm-level analytics-adoption measure.
- `rbi_nbfc_registry_clean.csv`: NBFC entity reference/classification records.
- Product mix/tenure files: supporting descriptive analysis.
