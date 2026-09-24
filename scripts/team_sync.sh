#!/usr/bin/env bash
# Share expensive artifacts (embeddings, OOF/test preds, folds) through the team S3 bucket.
# Code goes through git; big files go through here.
#
#   bash scripts/team_sync.sh push s3://amlc26-<team>      # upload data/features, oof/, experiments.csv
#   bash scripts/team_sync.sh pull s3://amlc26-<team>      # download what teammates pushed
#   bash scripts/team_sync.sh raw  s3://amlc26-<team>      # pull the raw dataset too (data/raw)
#
# Uses AWS_PROFILE (default: amlc). Teammates need the bucket policy from
# Context/aws-builder-center-free-tier-instructions.md. Never add --delete here.
set -euo pipefail
cd "$(dirname "$0")/.."
export AWS_PROFILE="${AWS_PROFILE:-amlc}"
MODE="${1:?push|pull|raw}"
BUCKET="${2:?s3://bucket-name}"
BUCKET="${BUCKET%/}"

case "$MODE" in
  push)
    aws s3 sync data/features "$BUCKET/features"
    aws s3 sync oof "$BUCKET/oof"
    [[ -f experiments.csv ]] && aws s3 cp experiments.csv "$BUCKET/experiments/$(whoami)_$(hostname -s).csv"
    ;;
  pull)
    aws s3 sync "$BUCKET/features" data/features
    aws s3 sync "$BUCKET/oof" oof
    aws s3 sync "$BUCKET/experiments" experiments_team
    ;;
  raw)
    aws s3 sync "$BUCKET/raw" data/raw
    ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
echo "sync $MODE done"
