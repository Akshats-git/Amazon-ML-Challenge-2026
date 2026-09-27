"""Replace the France rows of a base matching file with a France pairs table (s1_row, q_row); US/India rows untouched.
Caps: <=5 S2 / <=6 S3 per S1, keeping base links first. usage: patch_fr.py BASE_TSV FR_PAIRS OUT_TSV"""
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
base_tsv, frp, out = sys.argv[1:4]
s1=pl.read_parquet(io.norm_path('test',1),columns=['entity_id','country']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
q=io.load_norm('test','q',columns=['entity_id']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32))
n_s2=pl.scan_parquet(io.norm_path('test',2)).select(pl.len()).collect().item()
b=pl.read_csv(base_tsv,separator='\t',quote_char=None,infer_schema=False).with_columns(pl.col('matched_entity_ids').fill_null(''))
assert b.height==s1.height
ex=lambda df: df.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='')
bp=ex(b).join(s1.select(pl.col('entity_id').alias('source1_entity_id'),'s1_row','country'),on='source1_entity_id').join(q.select(pl.col('entity_id').alias('matched_entity_ids'),'q_row'),on='matched_entity_ids')
bfr=bp.filter(pl.col('country')=='France').select('s1_row','q_row')
fp=pl.read_parquet(frp).select('s1_row','q_row').unique()
isfr=s1.filter(pl.col('country')=='France').select('s1_row')
assert fp.join(isfr,on='s1_row',how='anti').height==0, 'non-France S1 in pairs'
assert fp['q_row'].n_unique()==fp.height, 'query linked twice'
# caps, base links first
fp=fp.join(bfr.with_columns(inb=pl.lit(1)),on=['s1_row','q_row'],how='left').with_columns(inb=pl.col('inb').fill_null(0), s3=pl.col('q_row')>=n_s2)
fp=fp.sort(['s1_row','s3','inb','q_row'],descending=[False,False,True,False]).with_columns(r=pl.int_range(pl.len()).over('s1_row','s3'))
capped=fp.filter(pl.col('r')>=pl.when(pl.col('s3')).then(6).otherwise(5))
fp=fp.filter(pl.col('r')<pl.when(pl.col('s3')).then(6).otherwise(5)).select('s1_row','q_row')
print(f'France pairs {fp.height:,} (capped {capped.height})  base France {bfr.height:,}  removed {bfr.join(fp,on=["s1_row","q_row"],how="anti").height:,}  added {fp.join(bfr,on=["s1_row","q_row"],how="anti").height:,}')
# every France query must not also be linked in US/India rows (queries are country-specific, but check)
other=bp.filter(pl.col('country')!='France').select('q_row')
assert fp.join(other,on='q_row',how='semi').height==0
agg=fp.join(s1.select('s1_row',pl.col('entity_id').alias('sid')),on='s1_row').join(q.select('q_row',pl.col('entity_id').alias('cid')),on='q_row').group_by('sid').agg(pl.col('cid').sort().str.join(',').alias('fr'))
b=b.join(s1.select(pl.col('entity_id').alias('source1_entity_id'),'country'),on='source1_entity_id',how='left',maintain_order='left').join(agg.rename({'sid':'source1_entity_id'}),on='source1_entity_id',how='left',maintain_order='left')
b=b.with_columns(matched_entity_ids=pl.when(pl.col('country')=='France').then(pl.col('fr').fill_null('')).otherwise(pl.col('matched_entity_ids')))
for c in ('US','India','France'):
    sb=b.filter(pl.col('country')==c); n=sb['matched_entity_ids'].str.split(',').list.eval(pl.element().filter(pl.element()!='')).list.len()
    print(f'{c}: S1 {sb.height:,} links {n.sum():,} links/S1 {n.mean():.4f} empty {(n==0).mean():.4f}')
os.makedirs(os.path.dirname(out),exist_ok=True)
b.select('source1_entity_id','matched_entity_ids').write_csv(out,separator='\t',quote_style='never',null_value='')
# sanity: US/India rows byte-identical to base
o=pl.read_csv(out,separator='\t',quote_char=None,infer_schema=False).with_columns(pl.col('matched_entity_ids').fill_null(''))
bb=pl.read_csv(base_tsv,separator='\t',quote_char=None,infer_schema=False).with_columns(pl.col('matched_entity_ids').fill_null(''))
cc=s1.select(pl.col('entity_id').alias('source1_entity_id'),'country')
d=o.join(bb,on='source1_entity_id',suffix='_b').join(cc,on='source1_entity_id').filter(pl.col('matched_entity_ids')!=pl.col('matched_entity_ids_b'))
print('changed S1 rows by country:', d.group_by('country').len().to_dicts(), ' same order:', (o['source1_entity_id']==bb['source1_entity_id']).all())
