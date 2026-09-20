#!/usr/bin/env bash
# Downloads PAMAP2 into ./data, which is gitignored. The dataset is ~656 MB;
# GitHub rejects files over 100 MB, so it never belongs in the repo.
#
# PAMAP2 Physical Activity Monitoring, Reiss & Stricker.
# UCI Machine Learning Repository. Licensed CC BY 4.0.
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
ARCHIVE="pamap2.zip"
URL="https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"

mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/PAMAP2_Dataset" ]; then
  echo "Dataset already present at $DATA_DIR/PAMAP2_Dataset — nothing to do."
  exit 0
fi

echo "Downloading PAMAP2 (~656 MB)..."
curl -fSL --retry 3 -o "$DATA_DIR/$ARCHIVE" "$URL"

echo "Extracting..."
unzip -q -o "$DATA_DIR/$ARCHIVE" -d "$DATA_DIR"

# UCI ships a zip containing another zip.
if [ -f "$DATA_DIR/PAMAP2_Dataset.zip" ]; then
  unzip -q -o "$DATA_DIR/PAMAP2_Dataset.zip" -d "$DATA_DIR"
fi

rm -f "$DATA_DIR/$ARCHIVE"
echo "Done. Protocol files: $DATA_DIR/PAMAP2_Dataset/Protocol/"
