"""Name-twin expansion retrieval: for target queries (unlinked by the base decision, address non-empty), candidate S1 = every
partition S1 whose core name (legal forms removed) equals the query's; ranked by IDF-weighted shared address tokens; top K new pairs.
usage: blk3.py PART COUNTRY [K=10]"""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os, sys, time, glob, numpy as np, polars as pl
os.chdir(ROOT); pass
from ber import io
from ber.pools import load_partition

part,C=sys.argv[1],sys.argv[2]; K=int(sys.argv[3]) if len(sys.argv)>3 else 10
split='test' if part=='test' else 'train'
t0=time.time()
LEG={'India':['pvt','ltd','private','limited','llp','opc','lp','public','co','company'],'US':['llc','inc','corp','co','ltd','lp','llp','pllc','pc','company','corporation','incorporated','limited','lc']}[C]
s1_rows,q_rows=load_partition(part,C)
# target queries = those rescue_feats uses: read from its output if present, else recompute from rf file
tq=pl.read_parquet(f'{R}/rf_{part}_{C}.parquet',columns=['q_row']).unique()
qn=io.load_norm(split,'q',columns=['name_norm','addr_norm']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32)).join(tq,on='q_row',how='semi')
sn=io.load_norm(split,'s1',columns=['name_norm','addr_norm']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32)).filter(pl.col('s1_row').is_in(pl.Series(s1_rows).implode()))
core=lambda c: pl.col(c).str.split(' ').list.eval(pl.element().filter(~pl.element().is_in(LEG)&(pl.element()!=''))).list.sort().list.join(' ')
qn=qn.with_columns(k=core('name_norm')).filter(pl.col('k')!=''); sn=sn.with_columns(k=core('name_norm'))
pairs=qn.select('q_row','k').join(sn.select('s1_row','k'),on='k')
print(f'{part} {C}: target q {tq.height:,}  core-name pairs {pairs.height:,}  ({time.time()-t0:.0f}s)',flush=True)
# exclude pairs already in original or second-retrieval candidates
old=pl.read_parquet(f'work/cands/{part}_{C}.parquet',columns=['q_row','s1_row']) if os.path.exists(f'work/cands/{part}_{C}.parquet') else pl.concat([pl.read_parquet(f,columns=['q_row','s1_row']) for f in sorted(glob.glob(f'work/feats/{part}/{C}_part-*.parquet'))])
b2=pl.read_parquet(f'{R}/b2_a_{part}_{C}.parquet',columns=['q_row','s1_row'])
pairs=pairs.join(old,on=['q_row','s1_row'],how='anti').join(b2,on=['q_row','s1_row'],how='anti')
# idf-weighted address overlap
N=sn.height
tok=sn.select('s1_row',t=pl.col('addr_norm').str.split(' ')).explode('t').filter(pl.col('t')!='').unique()
idf=tok.group_by('t').len().with_columns(idf=(np.log((1+N)/(1+pl.col('len')))+1).cast(pl.Float32)).select('t','idf')
qt=qn.select('q_row',t=pl.col('addr_norm').str.split(' ')).explode('t').filter(pl.col('t')!='').unique().join(idf,on='t',how='left').with_columns(pl.col('idf').fill_null(float(np.log(1+N)+1)))
x=pairs.select('q_row','s1_row').join(qt,on='q_row').join(tok.with_columns(ins=pl.lit(1)),on=['s1_row','t'],how='inner')
sc=x.group_by('q_row','s1_row').agg(blk_score=pl.col('idf').sum())
pairs=pairs.select('q_row','s1_row').join(sc,on=['q_row','s1_row'],how='left').with_columns(pl.col('blk_score').fill_null(0).cast(pl.Float32))
pairs=pairs.sort(['q_row','blk_score'],descending=[False,True]).with_columns(blk_rank=(pl.int_range(pl.len()).over('q_row')+1)).filter(pl.col('blk_rank')<=K).with_columns(pl.col('blk_rank').cast(pl.Int8))
pairs.write_parquet(f'{R}/b3_{part}_{C}.parquet')
print(f'  kept {pairs.height:,} new pairs  ({time.time()-t0:.0f}s)',flush=True)
if split=='train':
    gt=io.gt_rows()
    truth=gt.filter(pl.col('s1_row').is_in(pl.Series(s1_rows).implode())&pl.col('q_row').is_in(pl.Series(q_rows).implode()))
    bo=truth.join(old,on=['q_row','s1_row'],how='anti').join(b2,on=['q_row','s1_row'],how='anti')
    rec=bo.join(pairs,on=['q_row','s1_row'],how='semi')
    print(f'  blocked-out not in b2: {bo.height:,}  recovered by b3: {rec.height:,}')
