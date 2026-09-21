#!/usr/bin/env bash
# Uploads the PAMAP2 protocol files into a Unity Catalog volume.
#
# Scripted rather than drag-and-drop in the UI so the ingest path is
# reproducible: anyone with the repo and a PAT lands identical raw data.
#
# Prerequisites:
#   brew tap databricks/tap && brew install databricks
#   databricks configure --token     # host + PAT
#
# Create the volume first (see databricks/00_setup.sql).
set -euo pipefail

CATALOG="${CATALOG:-workspace}"
SCHEMA="${SCHEMA:-pamap2}"
VOLUME="${VOLUME:-raw}"
SRC="${SRC:-data/PAMAP2_Dataset/Protocol}"

TARGET="dbfs:/Volumes/${CATALOG}/${SCHEMA}/${VOLUME}/protocol"

if [ ! -d "$SRC" ]; then
  echo "No data at $SRC — run 'make data' first." >&2
  exit 1
fi

echo "Uploading $(ls "$SRC"/*.dat | wc -l | tr -d ' ') files to ${TARGET} ..."
databricks fs cp --overwrite --recursive "$SRC" "$TARGET"
echo "Done."
