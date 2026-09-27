"""France rows = R06 (x1+x2) decisions with three LB-evidenced class edits at equal house numbers:
  (a) drop links whose name edit swaps / adds a category word (Type-B same-address distractor) unless another confident
      record of the S1 carries the query's new word (supE_max > 0);
  (b) add the true-noise-word (groupe/france/developement/cie/services/fils/...) links x3 made that R06 did not;
  (c) add acronym pairs (query name = initials of the S1 name).
Writes a France pairs table (s1_row, q_row) and prints the per-class diff. usage: fr_rules.py [a,b,c]"""
import os as _os, sys as _sys
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
_os.makedirs(SP, exist_ok=True)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os
import polars as pl
os.chdir(ROOT); 
from ber import io

RULES = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else {'a', 'b', 'c'}
s1 = pl.read_parquet('work/norm/test_s1.parquet', columns=['entity_id', 'country']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
q = io.load_norm('test', 'q', columns=['entity_id']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32))
fr = s1.filter(pl.col('country') == 'France')


def pairs(d):
    m = pl.read_csv(f'{d}/matching_results.tsv', separator='\t', quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    m = m.join(fr.select(pl.col('entity_id').alias('source1_entity_id'), 's1_row'), on='source1_entity_id')
    m = m.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids') != '')
    return m.join(q.rename({'entity_id': 'matched_entity_ids'}), on='matched_entity_ids').select('s1_row', 'q_row')


base = pairs(os.environ.get('FR_BASE', 'output'))  # France rows of `BLEND_TAGS_UNSEEN=x1,x2 blend --tags x3 x4` (= R06 x1+x2)
x3 = pairs(os.environ.get('FR_X3', 'output_x3')) if ({'b', 'e'} & RULES) else None  # x3's own output, for edits (b)/(e)
cls = pl.read_parquet(f'{SP}/fr_cls.parquet')
sup = pl.read_parquet(f'{SP}/f12_test_France.parquet', columns=['q_row', 's1_row', 'supE_max', 'cls_c'])
cls = cls.join(sup, on=['q_row', 's1_row'], how='left')
out = base
if 'a' in RULES:
    bad = cls.filter((pl.col('num') == 'eq') & pl.col('cls').is_in(['cat_swap', 'cat_added']) & (pl.col('supE_max').fill_null(0) <= 0))
    out = out.join(bad.select('s1_row', 'q_row'), on=['s1_row', 'q_row'], how='anti')
if 'd' in RULES:
    # +GEN house-number shift vs S1 AND vs the consensus of the S1's other confident records (lone Type-A distractor);
    # US OOF: 2.15% true in this cell. cls_c == 3 is the +GEN class of stack2.numcls
    gen = cls.filter((pl.col('num') == 'gen') & (pl.col('cls_c') == 3))
    out = out.join(gen.select('s1_row', 'q_row'), on=['s1_row', 'q_row'], how='anti')
if 'b' in RULES:
    tn = cls.filter((pl.col('num') == 'eq') & (pl.col('cls') == 'TN_in')).select('s1_row', 'q_row')
    add = x3.join(tn, on=['s1_row', 'q_row'], how='semi').join(out, on=['s1_row', 'q_row'], how='anti')
    out = pl.concat([out, add])
if 'e' in RULES:
    # (b) restricted to pairs the large cross-encoder backs: x3's equal-address true-noise-word links with CE > 0.72
    ce_ = None
    for f in (0, 1):
        x = pl.read_parquet(f'{SP}/ceout/xl_P{f}_test_France.parquet').rename({'ce': f'c{f}'})
        ce_ = x if ce_ is None else ce_.join(x, on=['q_row', 's1_row'])
    ce_ = ce_.with_columns(pce=1 / (1 + (-(pl.col('c0') + pl.col('c1')) / 2).exp())).filter(pl.col('pce') > 0.72).select('s1_row', 'q_row')
    tn = cls.filter((pl.col('num') == 'eq') & (pl.col('cls') == 'TN_in')).select('s1_row', 'q_row')
    add = x3.join(tn, on=['s1_row', 'q_row'], how='semi').join(ce_, on=['s1_row', 'q_row'], how='semi').join(out, on=['s1_row', 'q_row'], how='anti')
    out = pl.concat([out, add])
if 'c' in RULES:
    ac = cls.filter((pl.col('num') == 'eq') & (pl.col('cls') == 'acronym')).select('s1_row', 'q_row')
    # the query must not already be linked elsewhere (argmax pairs, so at most one S1 per query)
    add = ac.join(out, on=['s1_row', 'q_row'], how='anti').join(out.select('q_row'), on='q_row', how='anti')
    out = pl.concat([out, add])
out = out.unique()
# per-source caps (<= 5 S2, <= 6 S3 per S1)
n_s2 = pl.scan_parquet(io.norm_path('test', 2)).select(pl.len()).collect().item()
cap = out.with_columns(s3=pl.col('q_row') >= n_s2).group_by('s1_row', 's3').len()
print('caps exceeded:', cap.filter(pl.when(pl.col('s3')).then(pl.col('len') > 6).otherwise(pl.col('len') > 5)).height)
rem = base.join(out, on=['s1_row', 'q_row'], how='anti'); addd = out.join(base, on=['s1_row', 'q_row'], how='anti')
lab = lambda d: d.join(cls, on=['s1_row', 'q_row'], how='left').group_by('num', 'cls').len().sort('len', descending=True)
print(f'France links: base {base.height:,}  new {out.height:,}  removed {rem.height:,}  added {addd.height:,}')
print('removed by class', lab(rem)); print('added by class', lab(addd))
tag = ''.join(sorted(RULES))
out.write_parquet(f'{SP}/fr_pairs_{tag}.parquet')
print('written', f'fr_pairs_{tag}.parquet')
