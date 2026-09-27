"""Attach cross-encoder logits to stage-3 feature tables. Pools get the OOF score of the CE trained on the other pool;
test gets the mean of both CEs. The score is kept only for rows in the band the CE was built for (LO < p < HI), so OOF and
test see it on the same rows. usage: add_ce.py FIN FOUT [countries]  (e.g. f fce US,India / f12 f12ce US,India,France)"""
import os as _os, sys as _sys
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
import os, sys
import numpy as np, polars as pl
CE = _os.path.join(SP, 'ceout')
fin, fout = sys.argv[1], sys.argv[2]
cs = sys.argv[3].split(',') if len(sys.argv) > 3 else ['US', 'India']
TAG = os.environ.get('CE_TAG', 'xr')
COL = os.environ.get('CE_COL', 'ce')
LO, HI = 0.003, 0.997
for c in cs:
    parts = ['test'] if c == 'France' else ['P0', 'P1', 'test']
    for part in parts:
        f = pl.read_parquet(f'{SP}/{fin}_{part}_{c}.parquet')
        if part == 'test':
            a = pl.read_parquet(f'{CE}/{TAG}_P0_test_{c}.parquet').rename({'ce': 'c0'})
            b = pl.read_parquet(f'{CE}/{TAG}_P1_test_{c}.parquet').rename({'ce': 'c1'})
            ce = a.join(b, on=['q_row', 's1_row']).select('q_row', 's1_row', ce=(pl.col('c0') + pl.col('c1')) / 2)
        else:
            other = 'P1' if part == 'P0' else 'P0'
            ce = pl.read_parquet(f'{CE}/{TAG}_{other}_{part}_{c}.parquet')
        ce = ce.unique(['q_row', 's1_row'])
        f = f.join(ce.rename({'ce': COL}), on=['q_row', 's1_row'], how='left')
        f = f.with_columns(pl.when((pl.col('p') > LO) & (pl.col('p') < HI)).then(pl.col(COL)).otherwise(None).alias(COL))
        if 'r' in f.columns:  # top-2 tables: the competing row's score (null when it has none)
            nn = pl.col(COL).is_not_null().sum().over('q_row')
            tot = pl.col(COL).sum().over('q_row')
            f = f.with_columns(pl.when(pl.col(COL).is_not_null() & (nn == 2)).then(tot - pl.col(COL))
                               .when(pl.col(COL).is_null() & (nn == 1)).then(tot).otherwise(None).alias(f'o_{COL}'))
        f.write_parquet(f'{SP}/{fout}_{part}_{c}.parquet')
        print(fout, part, c, f.height, 'with', COL, f[COL].is_not_null().sum())
