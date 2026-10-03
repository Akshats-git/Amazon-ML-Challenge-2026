"""(R20p, final) France additions and extra removals on top of the R20l France pairs ($R/fr_pairs_R3.parquet).
Inputs: France argmax table $R/frg.parquet (r20_fr_table.py), structural cells $R/cells_table.parquet and
$R/fr_cells.parquet (r20_cells.py), the France stage-3 model's test scores $SP/pfr_test.parquet and thresholds
$SP/pfr_thresholds.json (stage3.py fit with FPFX=f12ce TEST_C=France OUT=pfr).
Adds: argmax pairs that s8b leaves unlinked but the France stage-3 decision links, with stage-3 p3 >= 0.95 and large-CE
probability >= 0.95, outside the category, true-noise-word and pure generator-word classes and +GEN shifts, and outside
the add cells of the structural-transfer package (upload 1, lower on the LB); a query that is already linked is never
added. On US/India OOF, x1+x2 p < 0.5 with stage-3 p3 >= 0.95 is 99.4% true.
Removals: current links (x1+x2 p >= 0.5) of the R3 kinds that the large CE rejects (< 0.1) and stage 3 scores < 0.9.
Writes $R/fr_pairs_R3P.parquet. LB 0.989507 (R20l without this step: 0.989472)."""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import json, polars as pl

# France stage-3 decision on the argmax rows: per S1, the first link needs p3 >= T1 and later links p3 >= T2
th = json.load(open(f'{SP}/pfr_thresholds.json'))
g = pl.read_parquet(f'{R}/frg.parquet').with_columns(pl.col('L').cast(pl.Boolean)).join(
    pl.read_parquet(f'{SP}/pfr_test.parquet').select('q_row', 's1_row', 'p3'), on=['q_row', 's1_row'])
g = g.with_columns(rk=pl.col('p3').rank('ordinal', descending=True).over('s1_row'))
g = g.with_columns(D=pl.when(pl.col('rk') == 1).then(pl.col('p3') >= th['T1']).otherwise(pl.col('p3') >= th['T2']))

# add cells of the structural-transfer package (cells both US and India put at >= 93% true on the same street)
m = pl.read_parquet(f'{R}/cells_table.parquet')
good = (pl.col('n_US') >= 200) & (pl.col('n_India') >= 200) & (pl.col('agree') <= 0.06) & pl.col('cell').str.contains(r'\|S\|')
add_cells = m.filter(good & (pl.col('t_US') >= 0.93) & (pl.col('t_India') >= 0.93))
c = pl.read_parquet(f'{R}/fr_cells.parquet')
p1 = c.filter(pl.col('cell').is_in(add_cells['cell'].implode()) & (pl.col('L') == 0)).select('s1_row', 'q_row').with_columns(inP1=pl.lit(True))

# additions
add = g.filter(~pl.col('L') & pl.col('D'))
add = add.join(p1, on=['s1_row', 'q_row'], how='left').with_columns(pl.col('inP1').fill_null(False))
a = add.filter((pl.col('p3') >= 0.95) & (pl.col('xl') >= 0.95) & ~pl.col('cls').is_in(['cat_swap', 'cat_added', 'TN_in', 'pure_gen_word'])
               & (pl.col('num') != 'gen'))
a = a.filter(~pl.col('inP1')).select('s1_row', 'q_row')

# removals: the R3 pattern (R3 kinds, large CE < 0.1) among the current links, where stage 3 is below 0.9
base = pl.read_parquet(f'{R}/fr_pairs_R3.parquet').select('s1_row', 'q_row')
cur = g.join(base.with_columns(cur=pl.lit(True)), on=['s1_row', 'q_row'], how='left').filter(pl.col('cur').fill_null(False))
cur = cur.filter(pl.col('p') >= 0.5).with_columns(kind=pl.col('num') + '_' + pl.col('cls'))  # p < 0.5: edit-added links
R3K = ['eq_same_core', 'eq_rest', 'other_same_core', 'na_same_core', 'eq_typo', 'eq_cat_swap', 'eq_drop_only', 'gen_same_core',
       'na_cat_swap', 'other_rest', 'na_rest', 'other_cat_swap', 'eq_pure_gen_word', 'other_typo', 'other_drop_only', 'gen_typo',
       'eq_cat_added', 'na_typo', 'na_drop_only', 'gen_rest', 'gen_cat_swap', 'gen_drop_only', 'other_pure_gen_word',
       'gen_pure_gen_word', 'na_cat_added', 'other_cat_added']
ext = cur.filter(pl.col('kind').is_in(R3K) & (pl.col('xl') < 0.1) & (pl.col('p3') < 0.9)).select('s1_row', 'q_row')

# compose: never add a query that is already linked in France
a = a.join(base.select('q_row'), on='q_row', how='anti')
new = pl.concat([base.join(ext, on=['s1_row', 'q_row'], how='anti'), a]).unique()
assert new['q_row'].n_unique() == new.height, 'query linked twice'
print(f'R3 pairs {base.height:,}  removals {base.join(ext, on=["s1_row", "q_row"]).height:,}  additions {a.height:,}  new {new.height:,}')
new.write_parquet(f'{R}/fr_pairs_R3P.parquet')
