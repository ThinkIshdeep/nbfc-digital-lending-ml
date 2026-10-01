# Security

This repository is designed for public aggregate data and academic/portfolio use. It intentionally contains no borrower-level personal information, credentials, API keys, database passwords, or private NBFC records.

Do not add secrets to the repository. Use environment variables or a secret manager for deployment credentials.

The API is an analytical demo service. It is not hardened for internet-exposed production use and should be deployed behind standard authentication, rate limiting, TLS termination, logging, and network controls if it is ever extended beyond local/project use.
