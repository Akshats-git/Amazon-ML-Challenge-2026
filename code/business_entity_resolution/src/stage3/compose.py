"""Submission from stage-3 test scores: US/India decided on p3 (tie filter on the stage-2 blend's p/p2nd, thresholds
tuned on pooled OOF p3, per-source caps); every other S1 row (France) copied verbatim from a base file.
usage: compose.py OUT_DIR [BASE_DIR=output_hyb34] [T1 T2]
env: P3=<prefix of {P3}_test.parquet / {P3}_thresholds.json> (default p3); FR_PAIRS=<parquet of France (s1_row, q_row)>"""
import os as _os, sys as _sys
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
_os.makedirs(SP, exist_ok=True)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, os, sys
import numpy as np, polars as pl
os.chdir(ROOT); 
from ber import io
from ber.decide import assign, decide
out_dir = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else 'output_hyb34'
P3 = os.environ.get('P3', 'p3')
th = json.load(open(f'{SP}/{P3}_thresholds.json'))
T1, T2 = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (th['T1'], th['T2'])
print('thresholds', T1, T2)
n_s2 = pl.scan_parquet(io.norm_path('test', 2)).select(pl.len()).collect().item()
if os.path.exists(f'{SP}/{P3}_test_rows.parquet'):  # top-2 re-rank output: per query the higher-scoring of its two rows
    t = pl.read_parquet(f'{SP}/{P3}_test_rows.parquet').sort(['q_row', 'p5'], descending=[False, True])
    top = t.group_by('q_row', maintain_order=True).agg(pl.col('s1_row').first(), p=pl.col('p5').first(), second=pl.col('p5').slice(1, 1).first())
    top = top.filter(pl.col('second').is_null() | ((pl.col('p') - pl.col('second')) >= 1e-6)).drop('second')
else:
    t = pl.read_parquet(f'{SP}/{P3}_test.parquet')
    tiecol = 'p_s2' if 'p_s2' in t.columns else 'p'
    top = assign(t.select('q_row', 's1_row', pl.col(tiecol).alias('p'), 'p2nd'))  # drop ties of the stage-2 blend, rank by p
    top = top.drop('rk').join(t.select('q_row', 'p3'), on='q_row').with_columns(p=pl.col('p3')).drop('p3')
top = top.with_columns(rk=pl.col('p').rank('ordinal', descending=True).over('s1_row'))
pred = decide(top, T1, T2, n_s2=n_s2)
s1 = pl.read_parquet(io.norm_path('test', 1), columns=['entity_id', 'country']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
q = io.load_norm('test', 'q', columns=['entity_id']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32))
pairs = pred.join(s1.select('s1_row', pl.col('entity_id').alias('s1_id')), on='s1_row').join(q.select('q_row', pl.col('entity_id').alias('cand_id')), on='q_row')
agg = pairs.group_by('s1_id').agg(pl.col('cand_id').sort().str.join(',').alias('new'))
b = pl.read_csv(f'{base}/matching_results.tsv', separator='\t', quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
b = b.join(s1.select(pl.col('entity_id').alias('source1_entity_id'), 'country'), on='source1_entity_id', how='left', maintain_order='left')
assert b['country'].null_count() == 0 and b.height == s1.height
b = b.join(agg.rename({'s1_id': 'source1_entity_id'}), on='source1_entity_id', how='left', maintain_order='left')
seen = b['country'].is_in(['US', 'India'])
FRP = os.environ.get('FR_PAIRS')
if FRP:
    fp = pl.read_parquet(FRP)
    # per-source caps on the edited France pairs: keep existing links first, then the rest, up to 5 S2 / 6 S3 per S1
    base_fr = set()
    fp = fp.with_columns(s3=pl.col('q_row') >= n_s2).with_columns(r=pl.col('q_row').rank('ordinal').over('s1_row', 's3'))
    fp = fp.filter(pl.col('r') <= pl.when(pl.col('s3')).then(6).otherwise(5)).select('s1_row', 'q_row')
    fpp = fp.join(s1.select('s1_row', pl.col('entity_id').alias('s1_id')), on='s1_row').join(q.select('q_row', pl.col('entity_id').alias('cand_id')), on='q_row')
    fagg = fpp.group_by('s1_id').agg(pl.col('cand_id').sort().str.join(',').alias('frnew'))
    b = b.join(fagg.rename({'s1_id': 'source1_entity_id'}), on='source1_entity_id', how='left', maintain_order='left')
    b = b.with_columns(matched_entity_ids=pl.when(pl.col('country') == 'France').then(pl.col('frnew').fill_null('')).otherwise(pl.col('matched_entity_ids')))
b = b.with_columns(matched_entity_ids=pl.when(seen).then(pl.col('new').fill_null('')).otherwise(pl.col('matched_entity_ids').fill_null('')))
for c in ('US', 'India', 'France'):
    sb = b.filter(pl.col('country') == c)
    n = sb['matched_entity_ids'].str.split(',').list.eval(pl.element().filter(pl.element() != '')).list.len()
    print(f'{c}: S1 {sb.height:,}  links {n.sum():,}  links/S1 {n.mean():.4f}  empty {(n == 0).mean():.4f}')
os.makedirs(out_dir, exist_ok=True)
b.select('source1_entity_id', 'matched_entity_ids').write_csv(f'{out_dir}/matching_results.tsv', separator='\t', quote_style='never', null_value='')
# diff vs base per country
old = pl.read_csv(f'{base}/matching_results.tsv', separator='\t', quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
def ex(df):
    return df.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids') != '')
o, n_ = ex(old), ex(b.select('source1_entity_id', 'matched_entity_ids'))
cc = s1.select(pl.col('entity_id').alias('source1_entity_id'), 'country')
for c in ('US', 'India', 'France'):
    oc = o.join(cc, on='source1_entity_id').filter(pl.col('country') == c); nc = n_.join(cc, on='source1_entity_id').filter(pl.col('country') == c)
    print(f'  diff {c}: removed {oc.join(nc, on=["source1_entity_id", "matched_entity_ids"], how="anti").height:,}  added {nc.join(oc, on=["source1_entity_id", "matched_entity_ids"], how="anti").height:,}')
print('written', f'{out_dir}/matching_results.tsv')
