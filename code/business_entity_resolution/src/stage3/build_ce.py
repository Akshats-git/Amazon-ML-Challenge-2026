"""Cross-encoder datasets: argmax rows with LO < p < HI (plus a SAMPLE share of confident rows on train pools), raw
name/address of both records as text. usage: build_ce.py PART C [GPFX]"""
import os as _os, sys as _sys
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
_os.makedirs(SP, exist_ok=True)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys
import numpy as np, polars as pl

os.chdir(ROOT)
part, c = sys.argv[1], sys.argv[2]
gp = sys.argv[3] if len(sys.argv) > 3 else 'g'
LO, HI, SAMPLE = 0.003, 0.997, 0.04
split = 'test' if part == 'test' else 'train'
cols = ['q_row', 's1_row', 'p', 'entity_id'] + (['ok'] if split == 'train' else [])
g = pl.read_parquet(f'{SP}/{gp}_{part}_{c}.parquet', columns=cols)
unc = (pl.col('p') > LO) & (pl.col('p') < HI)
if split == 'train':
    rnd = pl.Series(np.random.default_rng(1).random(g.height) < SAMPLE)
    g = g.with_columns(unc=unc).filter(pl.col('unc') | rnd)
else:
    g = g.filter(unc).with_columns(unc=pl.lit(True))
s1 = pl.read_parquet(f'work/norm/{split}_s1.parquet', columns=['entity_id']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
g = g.join(s1.rename({'entity_id': 'sid'}), on='s1_row')
R = f'data/raw/{split}'
kw = dict(separator='\t', quote_char=None, infer_schema=False)
rq = pl.concat([pl.scan_csv(f'{R}/{split}_source{s}.tsv', **kw).filter(pl.col('entity_id').is_in(g['entity_id'].implode())).collect() for s in (2, 3)])
r1 = pl.scan_csv(f'{R}/{split}_source1.tsv', **kw).filter(pl.col('entity_id').is_in(g['sid'].implode())).collect()
txt = lambda df: df.with_columns(t=pl.col('business_name').fill_null('') + ' | ' + pl.col('business_address').fill_null('')).select('entity_id', 't')
g = g.join(txt(rq).rename({'t': 'a'}), on='entity_id', how='left').join(txt(r1).rename({'entity_id': 'sid', 't': 'b'}), on='sid', how='left')
g = g.with_columns(pl.col('a').fill_null(''), pl.col('b').fill_null(''))
g.write_parquet(f'{SP}/ce_{part}_{c}.parquet')
print(part, c, g.height, 'uncertain', g['unc'].sum(), ('pos %.3f' % g['ok'].mean()) if split == 'train' else '')
print(g.head(3).select('a', 'b'))
