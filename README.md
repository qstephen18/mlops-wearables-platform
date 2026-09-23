# mlops-wearables-platform

End-to-end MLOps pipeline on Databricks and AWS over wearable sensor data:
ingestion with data quality assertions, windowed features with subject-level
evaluation splits, MLflow-tracked training, metric-gated promotion to the Unity
Catalog registry, alias-based batch inference with per-prediction lineage, and
drift monitoring.

Infrastructure is Terraform-managed with OIDC-federated CI.

**[Design doc](DESIGN.md)** — decisions, tradeoffs and known limitations.

## Layout

- `databricks/` — pipeline notebooks, 01 through 06
- `infra/` — Terraform: S3, IAM, OIDC, budgets, remote state
- `scripts/` — data download and volume upload
- `.github/workflows/` — plan on pull request
