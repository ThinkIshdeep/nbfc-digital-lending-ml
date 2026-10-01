# Data Provenance

## Primary sources used

### RBI Working Group on Digital Lending — Annex C

Official report page:
https://www.rbi.org.in/Scripts/PublicationReportDetails.aspx?UrlPage=&ID=1189

Digital-lending PDF:
https://rbidocs.rbi.org.in/rdocs/PublicationReport/Pdfs/DIGITALLENDINGF6A90CA76A9B4B3E84AA0EBD24B307F1.PDF

The supplied source heading refers to **62 NBFCs**, while the visible tabular respondent rows retained in the project are **61**. The repository uses the observable 61 rows and does not fabricate a missing respondent.

### RBI NBFC sectoral deployment of credit

https://www.rbi.org.in/Scripts/Data_Sectoral_Deployment_NBFC.aspx

The supporting dataset contains sector/date outstanding-credit observations. These are retained as contextual data and are not relabelled as disbursement flows.

### RBI Digital Payments Index

https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=59632

### RBI NBFC registry

https://www.rbi.org.in/Scripts/BS_NBFCList.aspx

## Local source layout

- `data/raw/rbi/` contains source snapshots supplied with the field-project package.
- `data/processed/` contains cleaned analytical derivatives.
- `artifacts/metrics/model_registry.json` records the SHA-256 hash of the core dataset used by the training run.

## Integrity rules

- Missing markers are not converted to zero.
- No synthetic respondent is added.
- Amount and count are retained in the source units.
- Average ticket is derived only when both amount and count are available.
- Aggregate contextual datasets are kept separate from the respondent-period ML table.
