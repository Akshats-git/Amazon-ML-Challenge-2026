"""(R20) f12 tables (x1+x2 argmax rows: US/India pools + test France) + out-of-fold CE logits (xr -> ce, xl -> ce_xl)."""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)

import polars as pl
CE = SP + '/ceout'
LO, HI = 0.003, 0.997
def ce(tag, part, c):
    if part == 'test':
        a = pl.read_parquet(f'{CE}/{tag}_P0_test_{c}.parquet').rename({'ce': 'c0'}); b = pl.read_parquet(f'{CE}/{tag}_P1_test_{c}.parquet').rename({'ce': 'c1'})
        return a.join(b, on=['q_row', 's1_row']).select('q_row', 's1_row', ce=(pl.col('c0') + pl.col('c1')) / 2).unique(['q_row', 's1_row'])
    return pl.read_parquet(f'{CE}/{tag}_{"P1" if part == "P0" else "P0"}_{part}_{c}.parquet').unique(['q_row', 's1_row'])
for part, c in [('P0', 'US'), ('P1', 'US'), ('P0', 'India'), ('P1', 'India'), ('test', 'France')]:
    f = pl.read_parquet(f'{SP}/f12_{part}_{c}.parquet')
    for tag, col in (('xr', 'ce'), ('xl', 'ce_xl')):
        f = f.join(ce(tag, part, c).rename({'ce': col}), on=['q_row', 's1_row'], how='left')
        f = f.with_columns(pl.when((pl.col('p') > LO) & (pl.col('p') < HI)).then(pl.col(col)).otherwise(None).alias(col))
    f.write_parquet(f'{SP}/f12ce_{part}_{c}.parquet')
    print(part, c, f.height, 'ce', f['ce'].is_not_null().sum(), 'ce_xl', f['ce_xl'].is_not_null().sum())
