"""France argmax table (R06 x1+x2 argmax rows) with class, s8b link flag (fr_pairs_acde) and large-CE probability -> R/frg.parquet."""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import polars as pl, sys, random

g=pl.read_parquet(f'{SP}/g12_test_France.parquet', columns=['q_row','s1_row','p','p2nd','name_norm','addr_norm','addr_nums','name_norm_s','addr_norm_s','addr_nums_s','entity_id'])
cls=pl.read_parquet(f'{SP}/fr_cls.parquet')
lk=pl.read_parquet(f'{SP}/fr_pairs_acde.parquet').with_columns(L=pl.lit(1))
ce=None
for f in (0,1):
    x=pl.read_parquet(f'{SP}/ceout/xl_P{f}_test_France.parquet').rename({'ce':f'c{f}'})
    ce=x if ce is None else ce.join(x,on=['q_row','s1_row'])
ce=ce.with_columns(xl=1/(1+(-(pl.col('c0')+pl.col('c1'))/2).exp())).select('q_row','s1_row','xl')
g=g.join(cls,on=['q_row','s1_row'],how='left').join(lk,on=['s1_row','q_row'],how='left').join(ce,on=['q_row','s1_row'],how='left').with_columns(pl.col('L').fill_null(0))
g.write_parquet(R+'/frg.parquet')
print('France argmax table', g.height, 'linked', int(g['L'].sum()))
