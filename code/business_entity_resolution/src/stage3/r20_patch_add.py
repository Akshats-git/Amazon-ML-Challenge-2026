"""Add (s1_row, q_row) links to a matching file for one country (queries must be unlinked); caps <=5 S2/<=6 S3 keep base first.
usage: patch_add.py BASE_TSV ADD_PARQUET OUT_TSV"""
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
base_tsv, addp, out = sys.argv[1:4]
s1=pl.read_parquet(io.norm_path('test',1),columns=['entity_id','country']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
q=io.load_norm('test','q',columns=['entity_id']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32))
n_s2=pl.scan_parquet(io.norm_path('test',2)).select(pl.len()).collect().item()
b=pl.read_csv(base_tsv,separator='\t',quote_char=None,infer_schema=False).with_columns(pl.col('matched_entity_ids').fill_null(''))
bp=b.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='')
bp=bp.join(s1.select(pl.col('entity_id').alias('source1_entity_id'),'s1_row'),on='source1_entity_id').join(q.select(pl.col('entity_id').alias('matched_entity_ids'),'q_row'),on='matched_entity_ids').select('s1_row','q_row')
add=pl.read_parquet(addp).select('s1_row','q_row').unique()
add=add.join(bp.select('q_row'),on='q_row',how='anti')   # never relink a linked query
assert add['q_row'].n_unique()==add.height
allp=pl.concat([bp.with_columns(inb=pl.lit(1)),add.with_columns(inb=pl.lit(0))]).with_columns(s3=pl.col('q_row')>=n_s2)
allp=allp.sort(['s1_row','s3','inb','q_row'],descending=[False,False,True,False]).with_columns(k=pl.int_range(pl.len()).over('s1_row','s3'))
keep=allp.filter(pl.col('k')<pl.when(pl.col('s3')).then(6).otherwise(5))
dropped_base=allp.filter((pl.col('k')>=pl.when(pl.col('s3')).then(6).otherwise(5))&(pl.col('inb')==1)).height
added=keep.filter(pl.col('inb')==0).select('s1_row','q_row')
print(f'additions requested {add.height:,}  kept {added.height:,}  (base links dropped by caps: {dropped_base})')
agg=keep.select('s1_row','q_row').join(s1.select('s1_row',pl.col('entity_id').alias('sid')),on='s1_row').join(q.select('q_row',pl.col('entity_id').alias('cid')),on='q_row').group_by('sid').agg(pl.col('cid').sort().str.join(',').alias('new'))
touched=added.select('s1_row').unique().join(s1.select('s1_row',pl.col('entity_id').alias('sid')),on='s1_row').select('sid')
b=b.join(agg.rename({'sid':'source1_entity_id'}),on='source1_entity_id',how='left',maintain_order='left')
b=b.with_columns(matched_entity_ids=pl.when(pl.col('source1_entity_id').is_in(touched['sid'].implode())).then(pl.col('new')).otherwise(pl.col('matched_entity_ids')))
os.makedirs(os.path.dirname(out),exist_ok=True)
b.select('source1_entity_id','matched_entity_ids').write_csv(out,separator='\t',quote_style='never',null_value='')
cc=s1.select(pl.col('entity_id').alias('source1_entity_id'),'country')
o=pl.read_csv(out,separator='\t',quote_char=None,infer_schema=False).with_columns(pl.col('matched_entity_ids').fill_null(''))
bb=pl.read_csv(base_tsv,separator='\t',quote_char=None,infer_schema=False).with_columns(pl.col('matched_entity_ids').fill_null(''))
dd=o.join(bb,on='source1_entity_id',suffix='_b').join(cc,on='source1_entity_id').filter(pl.col('matched_entity_ids')!=pl.col('matched_entity_ids_b'))
print('changed S1 rows by country:', dd.group_by('country').len().to_dicts(), ' same order:', (o['source1_entity_id']==bb['source1_entity_id']).all())
