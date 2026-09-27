#!/usr/bin/env bash
# Full pipeline, run from the directory holding data/ (or set DATA_DIR). Produces output/matching_results.tsv and
# output/candidate_pairs.tsv. On a 13 GB laptop:  N_JOBS=8 BLOCK_THREADS=4 bash code/business_entity_resolution/run_all.sh
#   START=<step> bash code/business_entity_resolution/run_all.sh    # resume from a step
set -euo pipefail
R="python code/business_entity_resolution/src/run.py"
mkdir -p work/logs
export PYTHONUNBUFFERED=1

STEPS=(env check1 check2 norm0 dicts norm pools block feats check8 xfeats nfeats x1_train1 x1_train2 x1_tune x1_predict
       x2_train1 x2_train2 x2_tune x2_predict blend12 x3_train1 x3_train2 x3_tune x3_predict
       x4_train1 x4_train2 x4_tune x4_predict blend)
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
    blend12)    USE_XFEATS=1 WRITE_CANDIDATES=0 OUTPUT_DIR=output_x1x2 $R blend --tags x1 x2 --check-ids ;;
    # x3: + 16 number features, all queries (~58.5M pairs per model: needs ~64 GB RAM, ~2.5 h on 32 vCPU)
    x3_*)       TAG=x3 USE_XFEATS=1 USE_NFEATS=1 TRAIN_QUERY_FRAC=1.0 MASK_LEX_FRAC=0.5 WRITE_CANDIDATES=0 OUTPUT_DIR=output_x3 \
                  $R "${s#x3_}" $([[ $s == x3_predict ]] && echo --check-ids) ;;
    # x4: x3's data and features, 255 leaves / min_child 200 / feature_fraction 0.6 / lambda_l2 5, seed 43
    x4_*)       TAG=x4 USE_XFEATS=1 USE_NFEATS=1 TRAIN_QUERY_FRAC=1.0 MASK_LEX_FRAC=0.5 WRITE_CANDIDATES=0 OUTPUT_DIR=output_x4 \
                  LGB1_PARAMS='{"num_leaves": 255, "min_child_samples": 200, "feature_fraction": 0.6, "lambda_l2": 5.0, "seed": 43}' \
                  LGB2_PARAMS='{"seed": 43}' $R "${s#x4_}" $([[ $s == x4_predict ]] && echo --check-ids) ;;
    # final: x3+x4 for countries seen in training; France (absent from train) keeps the x1+x2 blend (better on the LB)
    blend)      USE_XFEATS=1 WRITE_CANDIDATES=1 OUTPUT_DIR=output BLEND_TAGS_UNSEEN=x1,x2 $R blend --tags x3 x4 --check-ids ;;
    *)          $R "$s" ;;
  esac 2>&1 | tee -a "work/logs/full_${s}.log"
done
echo "=== all done $(date '+%F %T')" | tee -a work/logs/run_all.log
