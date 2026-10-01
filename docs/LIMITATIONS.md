# Limitations and Research Boundary

This repository demonstrates an **industry-style ML engineering workflow on a small public academic dataset**.

It should not be described as a validated production lending model because:

- the core table is a small panel with only five periods;
- respondent identities are anonymous in the cleaned table;
- internal data-analytics adoption is not directly measured;
- borrower-level credit risk and repayment outcomes are not present;
- the target is next **observed** disbursement, not loan approval quality or portfolio profitability;
- supporting RBI sector data have different units of observation and are therefore not row-wise joined into the ML table.

A stronger future system would require a much larger firm-time panel and borrower-level features such as application timestamp, income, credit score, bureau history, risk grade, approval outcome, processing time, repayment status, default flag and documented analytics/adoption events.
