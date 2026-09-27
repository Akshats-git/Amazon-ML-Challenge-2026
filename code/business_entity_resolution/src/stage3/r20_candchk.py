"""Check every matched France pair of a matching file is in work/cands/test_France.parquet (and US/India unchanged vs base)."""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import sys, os, polars as pl
os.chdir(ROOT); pass
from ber import io
m=sys.argv[1]
s1=pl.read_parquet(io.norm_path('test',1),columns=['entity_id','country']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
q=io.load_norm('test','q',columns=['entity_id']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32))
b=pl.read_csv(m,separator='\t',quote_char=None,infer_schema=False).with_columns(pl.col('matched_entity_ids').fill_null(''))
p=b.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='')
p=p.join(s1.select(pl.col('entity_id').alias('source1_entity_id'),'s1_row','country'),on='source1_entity_id').join(q.select(pl.col('entity_id').alias('matched_entity_ids'),'q_row'),on='matched_entity_ids')
for c in sys.argv[2].split(','):
    cand=pl.read_parquet(f'work/cands/test_{c}.parquet',columns=['q_row','s1_row'])
    pc=p.filter(pl.col('country')==c)
    miss=pc.join(cand,on=['q_row','s1_row'],how='anti')
    print(c,'pairs',pc.height,'not in candidates',miss.height)
print('queries linked twice:', p.group_by('q_row').len().filter(pl.col('len')>1).height)
