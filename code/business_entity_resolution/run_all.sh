#!/usr/bin/env bash
# Full pipeline, run from the directory holding data/ (or set DATA_DIR). Produces output/matching_results.tsv and
# output/candidate_pairs.tsv. On a 13 GB laptop:  N_JOBS=8 BLOCK_THREADS=4 bash code/business_entity_resolution/run_all.sh
#   START=<step> bash code/business_entity_resolution/run_all.sh    # resume from a step
set -euo pipefail
R="python code/business_entity_resolution/src/run.py"
mkdir -p work/logs
export PYTHONUNBUFFERED=1

STEPS=(env check1 check2 norm0 dicts norm pools block feats check8 xfeats x1_train1 x1_train2 x1_tune x1_predict
       x2_train1 x2_train2 x2_tune x2_predict blend)
START="${START:-env}"
started=0
for s in "${STEPS[@]}"; do
  [[ "$s" == "$START" ]] && started=1
  [[ $started == 1 ]] || continue
  echo "=== $s  $(date '+%F %T')" | tee -a work/logs/run_all.log
  case "$s" in
    norm0)      rm -f work/dicts/*.json; $R norm ;;   # first pass: no dictionaries yet
    x1_*)       TAG=x1 USE_XFEATS=1 TRAIN_QUERY_FRAC=0.15 OUTPUT_DIR=output_x1 $R "${s#x1_}" $([[ $s == x1_predict ]] && echo --check-ids) ;;
    x2_*)       TAG=x2 USE_XFEATS=1 TRAIN_QUERY_FRAC=0.30 X16=1 WRITE_CANDIDATES=0 OUTPUT_DIR=output_x2 \
                  $R "${s#x2_}" $([[ $s == x2_predict ]] && echo --check-ids) ;;
    blend)      USE_XFEATS=1 WRITE_CANDIDATES=1 OUTPUT_DIR=output $R blend --tags x1 x2 --check-ids ;;
    *)          $R "$s" ;;
  esac 2>&1 | tee -a "work/logs/full_${s}.log"
done
echo "=== all done $(date '+%F %T')" | tee -a work/logs/run_all.log
