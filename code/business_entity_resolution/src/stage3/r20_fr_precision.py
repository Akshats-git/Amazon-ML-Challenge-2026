"""(R20l) France precision removals. Inputs: France argmax table $R/frg.parquet (r20_fr_table.py: x1+x2 p, class num/cls,
s8b link flag L, large-CE probability xl), the France stage-3 model's test scores $SP/pfr_test.parquet (stage3.py fit with
FPFX=f12ce TEST_C=France OUT=pfr on the France-safe feature list) and the CE veto $R/fr_veto2.parquet (r20_fr_veto.py).
Removes s8b France links that the France stage-3 decision drops AND the large CE rejects (< 0.3 or unscored), except the
LB-backed classes (edit-added links p < 0.5, acronyms, true-noise-word appends and swaps, no-shared-word names), plus the
veto. Writes $R/fr_pairs_R3.parquet. LB: 0.989472 with the India/US rescue (s8b 0.987777)."""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import json, polars as pl
LEG = {'sarl','sas','sasu','eurl','sa','sci','snc','and','et','scop','gie','ste','cie','co','com','de','du','des','la','le','les','d','l','s','a'}
TN = {'groupe','france','developement','cie','services','fils','and','et','asocies','freres','compagnie'}
th = json.load(open(f'{SP}/pfr_thresholds.json'))
g = pl.read_parquet(f'{R}/frg.parquet').with_columns(pl.col('L').cast(pl.Boolean)).join(
    pl.read_parquet(f'{SP}/pfr_test.parquet').select('q_row', 's1_row', 'p3'), on=['q_row', 's1_row'])
g = g.with_columns(rk=pl.col('p3').rank('ordinal', descending=True).over('s1_row'))
g = g.with_columns(D=pl.when(pl.col('rk') == 1).then(pl.col('p3') >= th['T1']).otherwise(pl.col('p3') >= th['T2']))
rem = g.filter(pl.col('L') & ~pl.col('D'))
def miss(q, s):
    qs = set(q.split()); return sum(1 for t in s.split() if t and t not in LEG and t not in TN and t not in qs)
rem = rem.with_columns(miss=pl.Series([miss(q, s) for q, s in zip(rem['name_norm'].to_list(), rem['name_norm_s'].to_list())], dtype=pl.Int32))
keep = (pl.col('p') < 0.5) | pl.col('cls').is_in(['acronym', 'TN_in', 'no_shared'])
r3 = rem.filter(~keep & (pl.col('xl').is_null() | (pl.col('xl') < 0.3))).select('s1_row', 'q_row')
v = pl.read_parquet(f'{R}/fr_veto2.parquet').select('s1_row', 'q_row')
drop = pl.concat([r3, v]).unique()
base = pl.read_parquet(f'{SP}/fr_pairs_acde.parquet')
new = base.join(drop, on=['s1_row', 'q_row'], how='anti')
print(f'stage-3 drops {rem.height:,}; R3 {r3.height:,}; veto {v.height:,}; removed {base.height - new.height:,}')
new.write_parquet(f'{R}/fr_pairs_R3.parquet')
