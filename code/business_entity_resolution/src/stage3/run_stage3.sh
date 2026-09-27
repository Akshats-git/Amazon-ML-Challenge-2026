#!/usr/bin/env bash
# Final stage (R18/R19): top-2 re-rank with two cross-encoders for US/India + evidence-backed France edits.
# Prerequisites (README "Commands"): runs x1..x4, `OUTPUT_DIR=output_x3 run.py predict` (x3's own file, used by France
# edit e) and the final `BLEND_TAGS_UNSEEN=x1,x2 run.py blend --tags x3 x4`, which writes output/.
# Run from the data directory. Tables go to work/stage3 (S3_DIR). Steps 5-6 need a CUDA GPU (one A100 40GB:
# base member ~35 min, large member ~110 min, both folds in parallel incl. scoring).
set -euo pipefail
PY=${PY:-python}
D=$(dirname "$0")
export S3_DIR=${S3_DIR:-work/stage3}
CE=$S3_DIR/ce
mkdir -p "$CE" "$S3_DIR/ceout"

# 1. argmax tables (stage-2 blend x3+x4 for US/India; x1+x2 for France) and top-2 tables (argmax + runner-up rows)
for p in P0 P1 test; do for c in US India; do
  BLEND=x3,x4 PFX=g $PY $D/tables.py $p $c
  BLEND=x3,x4 PFX=h $PY $D/tables_top2.py $p $c
done; done
BLEND=x1,x2 PFX=g12 $PY $D/tables.py test France

# 2. top-2 features (consensus with the S1's other queries, raw-text perturbations, model disagreement, competing row)
for p in P0 P1 test; do for c in US India; do HPFX=h KPFX=k $PY $D/top2.py build $p $c; done; done
GPFX=g12 FPFX=f12 $PY $D/stage3.py build test France          # France consensus features for the France edits

# 3. cross-encoder text pairs: uncertain argmax rows (0.003 < p < 0.997, +4% of the rest on the pools) ...
for p in P0 P1 test; do for c in US India; do $PY $D/build_ce.py $p $c; done; done
$PY $D/build_ce.py test France g12
mv "$S3_DIR"/ce_*.parquet "$CE/"
# ... and plausible runner-up rows (p >= 0.01), appended to the scoring files (*_top2)
for p in P0 P1 test; do for c in US India; do $PY $D/build_ce2.py $p $c; done; done
CE_DIR=$CE $PY $D/ce_rows_top2.py

# 4. cross-encoder members: each fold trains on one pool's argmax file and scores the other pool + test
ce() {  # tag model batch lr score_suffix
  for f in P0 P1; do
    CE_DIR=$CE CE_TAG=$1 CE_MODEL=$2 CE_BS=$3 CE_EBS=$((8 * $3)) CE_LEN=128 CE_LR=$4 CE_BF16=1 CE_FREEZE_EMB=0 CE_NW=4 \
      CE_SCORE_SFX=$5 $PY $D/ce_train.py $f
  done
  mv "$CE"/out/$1_*.parquet "$S3_DIR/ceout/"
}
ce xr FacebookAI/xlm-roberta-base 128 3e-5 ""          # base: scores argmax rows
ce xl FacebookAI/xlm-roberta-large 64 1e-5 _top2        # large: scores argmax + runner-up rows

# 5. attach out-of-fold CE logits (test = mean of both folds) and fit the top-2 re-rank (cross-fitted P0 <-> P1)
CE_TAG=xl CE_COL=ce_xl $PY $D/add_ce.py k kce US,India
CE_TAG=xr CE_COL=ce_xr $PY $D/add_ce.py kce kce2 US,India
K=r,lp,lpo,dp,lx1,lx2,lx3,lx4,n_cand,is_india,n_conf_s,sum_p_s,n_q_s,rk_s,n_conf_s_same_src,n_same_nums,n_same_name,n_same_addr,cls_s,cls_c,s_eq_c,cn_c,nE,nM,n_shared,supE_min,supE_max,supE_any,supM_min,supM_max,supM_any,is_acr,q_same_name_all,s1_same_name,q_addr_empty,is_s3,d_dbl,q_tri,d_acc,d_sfx,d_frac,d_rep,q_brack,q_paren,q_hash,o_n_conf_s,o_sum_p_s,o_n_q_s,o_n_same_name,o_cls_s,o_nE,o_nM,o_n_shared,o_s1_same_name,o_supM_any,ce_xl,o_ce_xl,ce_xr,o_ce_xr
KPFX=kce2 OUT=p8b $PY $D/top2.py fit $K

# 6. France: x1+x2 decisions of the blend output with the evidence-backed class edits
$PY $D/fr_classes.py
FR_BASE=output FR_X3=output_x3 $PY $D/fr_rules.py a,c,d,e

# 7. write output/matching_results.tsv: US/India from the top-2 re-rank (per query the higher-scoring row; per S1 first
#    link >= T1, later >= T2; per-source caps), France from the edited pairs. Then validate (README).
P3=p8b FR_PAIRS="$S3_DIR/fr_pairs_acde.parquet" $PY $D/compose.py output output

# 8. (R20) India/US blocking rescue with a rescue cross-encoder. Work dir: R20_DIR (default $S3_DIR/r20).
#    The France structural transfer (r20_fr_transfer.py, +4,317 / -138 links) scored 0.987654 on the LB vs 0.987777
#    without it, so it is NOT applied; France gets only the precision removals of e).
R20=${R20_DIR:-$S3_DIR/r20}
#    a) candidates for queries left unlinked that have an address: (i) address-heavy second retrieval (name 0.1 /
#       address 0.8 / char 0.1, top 10, new pairs only), (ii) name-twin expansion (all S1 sharing the core name, ranked
#       by IDF-weighted shared address tokens, top 10); v2 LightGBM rescue scorer cross-fitted P0 <-> P1 (28 features)
for c in India US; do
  for p in P0 P1 test; do $PY $D/r20_block2.py $p $c 0.1,0.8,0.1 10 a; done
  for p in P0 P1 test; do BASE_TSV=output/matching_results.tsv $PY $D/r20_rescue_feats.py $p $c; done   # target queries
  for p in P0 P1 test; do $PY $D/r20_block3.py $p $c 10; done
  for p in P0 P1 test; do BASE_TSV=output/matching_results.tsv $PY $D/r20_rescue_feats.py $p $c; done   # + name twins
  $PY $D/r20_rescue_train.py $c
#    b) rescue cross-encoder sets: queries whose best v2 score >= 0.03, top 4 candidates each, raw "name | address" text
  $PY $D/r20_build_rce.py $c
done
#    c) GPU (A100 40GB, ~21 min per fold, both folds in parallel): xlm-roberta-large, 3 passes over one pool's rescue
#       candidates (US+India), scores the other pool + test
for f in P0 P1; do
  CE_DIR=$R20/rce CE_TAG=rxl CE_MODEL=FacebookAI/xlm-roberta-large CE_BS=32 CE_EBS=256 CE_LEN=128 CE_LR=1e-5 CE_BF16=1 \
    CE_FREEZE_EMB=0 CE_NW=4 CE_REP=3 $PY $D/r20_rce_train.py $f > $R20/rce/tr_$f.log 2>&1 &
done
wait
#    d) rescue v4 = v2 features + CE logit and its within-query rank / gap to the best / margin over the runner-up;
#       cross-fitted, tau tuned on OOF (India 0.99107 -> 0.99262 at 0.7; US 0.99210 -> 0.99237 at 0.85); links the best
#       new candidate of each target query when its score >= tau
$PY $D/r20_rescue4.py India
$PY $D/r20_rescue4.py US
#    e) France precision removals (R20l): France argmax table + large-CE probabilities, the CE veto (its inputs come from
#       the transfer tables, which are built but not applied), the France stage-3 model (stage 3 rebuilt on x1+x2 with
#       France-safe features; US/India OOF 0.99128), then drop s8b France links that stage 3 drops AND the large CE
#       rejects (< 0.3), except LB-backed classes, plus the veto: -3,164 France links
$PY $D/r20_fr_table.py
$PY $D/r20_cells.py
$PY $D/r20_initl.py
$PY $D/r20_fr_transfer.py
$PY $D/r20_fr_veto.py
$PY $D/r20_merge_fr.py
FR=lp,lp2,gap,lx1,lx2,n_cand,is_india,n_conf_s,sum_p_s,n_q_s,rk_s,n_conf_s_same_src,n_same_nums,n_same_name,n_same_addr,cls_s,cls_c,s_eq_c,cn_c,nE,nM,n_shared,supE_min,supE_max,supE_any,supM_min,supM_max,supM_any,is_acr,q_addr_empty,is_s3,d_dbl,q_tri,d_sfx,d_frac,d_rep,q_brack,q_paren,q_hash,ce,ce_xl
FPFX=f12ce TEST_C=France OUT=pfr $PY $D/stage3.py fit $FR
$PY $D/r20_fr_precision.py
$PY $D/r20_patch_fr.py output/matching_results.tsv "$R20/fr_pairs_R3.parquet" output_r20/fr/matching_results.tsv
$PY $D/r20_patch_add.py output_r20/fr/matching_results.tsv "$R20/rescue4_add_test_India.parquet" output_r20/in/matching_results.tsv
$PY $D/r20_patch_add.py output_r20/in/matching_results.tsv "$R20/rescue4_add_test_US.parquet" output_r20/final/matching_results.tsv
#    f) candidate_pairs.tsv = original blocking candidates U every rescue-scored pair (the rescue model runs inference on them)
$PY $D/r20_regen_cands.py output_r20/final/candidate_pairs.tsv output_r20/final/matching_results.tsv India,US
