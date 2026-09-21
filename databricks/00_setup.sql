-- Run once in a Databricks SQL editor or notebook cell.
-- Free Edition ships a `workspace` catalog; change CATALOG below if yours differs.

CREATE SCHEMA IF NOT EXISTS workspace.pamap2
  COMMENT 'PAMAP2 wearable activity data: bronze, silver and gold tables.';

-- Managed volume for raw files. Using a managed volume rather than an external
-- location on S3 because Free Edition does not expose the account UUID needed
-- to build a Unity Catalog storage credential. The tradeoff is deliberate:
-- Databricks owns the ML lifecycle, AWS owns IaC, CI/CD and cost governance.
CREATE VOLUME IF NOT EXISTS workspace.pamap2.raw
  COMMENT 'Raw PAMAP2 .dat files, uploaded by scripts/upload_to_volume.sh';
