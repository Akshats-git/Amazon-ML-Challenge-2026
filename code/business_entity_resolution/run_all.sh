#!/usr/bin/env bash
# Full pipeline on a 16 vCPU / 64 GB box, run from the directory holding data/ (or set DATA_DIR).
#   bash code/business_entity_resolution/run_all.sh            # everything
#   START=train1 bash code/business_entity_resolution/run_all.sh   # resume from a step
# Tier-0 outputs land in output_tier0/ (upload these first); the final outputs land in output/.
set -euo pipefail
R="python code/business_entity_resolution/src/run.py"
mkdir -p work/logs
export PYTHONUNBUFFERED=1

STEPS=(env check1 check2 norm0 dicts norm pools block feats check8 tier0 tier0_predict train1 train2 tune predict)
START="${START:-env}"
started=0
for s in "${STEPS[@]}"; do
  [[ "$s" == "$START" ]] && started=1
  [[ $started == 1 ]] || continue
  echo "=== $s  $(date '+%F %T')" | tee -a work/logs/run_all.log
  case "$s" in
    norm0)         rm -f work/dicts/*.json; $R norm ;;   # first pass: no dictionaries yet
    tier0)         $R train1 --tier0 ;;
    tier0_predict) OUTPUT_DIR=output_tier0 $R predict --tier0 --check-ids ;;
    train1)        $R train1 --reuse ;;            # reuses the tier-0 P0 model, fits P1
    predict)       $R predict --check-ids ;;
    *)             $R "$s" ;;
  esac 2>&1 | tee -a "work/logs/full_${s}.log"
done
echo "=== all done $(date '+%F %T')" | tee -a work/logs/run_all.log
