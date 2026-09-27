"""India rescue: features for NEW address-heavy candidates of queries the current decision leaves unlinked.
usage: rescue_feats.py PART (P0|P1|test) [COUNTRY=India]"""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os, sys, json, time, numpy as np, polars as pl
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler
os.chdir(ROOT); pass
from ber import io
from ber.pools import load_partition

part=sys.argv[1]; C=sys.argv[2] if len(sys.argv)>2 else 'India'
split='test' if part=='test' else 'train'
t0=time.time()
th=json.load(open(f'{SP}/p8b_thresholds.json')); T1,T2=th['T1'],th['T2']
s1_rows,q_rows=load_partition(part,C)
# base decision -> linked queries
if part=='test':
    s1=pl.read_parquet(io.norm_path('test',1),columns=['entity_id','country']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
    q=io.load_norm('test','q',columns=['entity_id']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32))
    b=pl.read_csv(_os.environ.get('BASE_TSV','output/matching_results.tsv'),separator='\t',quote_char=None,infer_schema=False).with_columns(pl.col('matched_entity_ids').fill_null(''))
    bp=b.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='')
    linked=bp.join(q.select(pl.col('entity_id').alias('matched_entity_ids'),'q_row'),on='matched_entity_ids').select('q_row').unique()
    top=pl.read_parquet(f'{SP}/p8b_test_rows.parquet').sort(['q_row','p5'],descending=[False,True]).group_by('q_row',maintain_order=True).agg(p5=pl.col('p5').first(),arg_s1=pl.col('s1_row').first())
else:
    r=pl.read_parquet(f'{SP}/p8b_oof_rows.parquet').filter(pl.col('pool')==part).sort(['q_row','p5'],descending=[False,True])
    top=r.group_by('q_row',maintain_order=True).agg(arg_s1=pl.col('s1_row').first(),p5=pl.col('p5').first(),second=pl.col('p5').slice(1,1).first())
    tt=top.filter(pl.col('second').is_null()|((pl.col('p5')-pl.col('second'))>=1e-6)).with_columns(rk=pl.col('p5').rank('ordinal',descending=True).over('arg_s1'))
    tt=tt.with_columns(link=pl.when(pl.col('rk')==1).then(pl.col('p5')>=T1).otherwise(pl.col('p5')>=T2))
    linked=tt.filter('link').select('q_row')
    top=top.select('q_row','p5','arg_s1')
new=pl.read_parquet(f'{R}/b2_a_{part}_{C}.parquet')
new=new.with_columns(src=pl.lit(2,pl.Int8))
if os.path.exists(f'{R}/b3_{part}_{C}.parquet') and os.environ.get('USE_B3','1')=='1':
    b3=pl.read_parquet(f'{R}/b3_{part}_{C}.parquet').with_columns(src=pl.lit(3,pl.Int8))
    new=pl.concat([new.select('q_row','s1_row',pl.col('blk_score').cast(pl.Float32),pl.col('blk_rank').cast(pl.Int8),'src'),
                   b3.select('q_row','s1_row',pl.col('blk_score').cast(pl.Float32),pl.col('blk_rank').cast(pl.Int8),'src')]).unique(['q_row','s1_row'])
qn=io.load_norm(split,'q',columns=['name_norm','addr_norm','addr_nums','name_sq']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32))
qn=qn.filter(pl.col('q_row').is_in(pl.Series(q_rows).implode()))
tq=qn.filter(pl.col('addr_norm')!='').join(linked,on='q_row',how='anti').select('q_row')
new=new.join(tq,on='q_row',how='semi')
print(f'{part} {C}: target queries {tq.height:,}  new pairs {new.height:,}  ({time.time()-t0:.0f}s)',flush=True)
s1n=io.load_norm(split,'s1',columns=['name_norm','addr_norm','addr_nums','name_sq']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
s1n=s1n.filter(pl.col('s1_row').is_in(pl.Series(s1_rows).implode()))
# twins: S1 count per name_norm in the partition
tw=s1n.group_by('name_norm').len().rename({'len':'tw'})
s1n=s1n.join(tw,on='name_norm',how='left')
# address token idf over partition S1
tok=s1n.select('s1_row',t=pl.col('addr_norm').str.split(' ')).explode('t').filter(pl.col('t')!='').unique()
N=s1n.height
idf=tok.group_by('t').len().with_columns(idf=(np.log((1+N)/(1+pl.col('len')))+1).cast(pl.Float32)).select('t','idf')
d=new.join(qn.rename({'name_norm':'qn','addr_norm':'qa','addr_nums':'qnum','name_sq':'qsq'}),on='q_row').join(s1n.rename({'name_norm':'sn','addr_norm':'sa','addr_nums':'snum','name_sq':'ssq'}),on='s1_row')
d=d.join(top.select('q_row','p5',pl.col('arg_s1')),on='q_row',how='left')
print('joined',d.height,f'({time.time()-t0:.0f}s)',flush=True)
def cp(scorer,a,b):
    return process.cpdist(d[a].to_list(),d[b].to_list(),scorer=scorer,workers=-1).astype(np.float32)
feats={}
feats['n_tset']=cp(fuzz.token_set_ratio,'qn','sn'); feats['n_tsort']=cp(fuzz.token_sort_ratio,'qn','sn'); feats['n_part']=cp(fuzz.partial_ratio,'qn','sn'); feats['n_r']=cp(fuzz.ratio,'qn','sn')
feats['sq_jw']=cp(JaroWinkler.normalized_similarity,'qsq','ssq')
feats['a_tset']=cp(fuzz.token_set_ratio,'qa','sa'); feats['a_part']=cp(fuzz.partial_ratio,'qa','sa')
print('rapidfuzz done',f'({time.time()-t0:.0f}s)',flush=True)
d=d.with_columns(**{k:pl.Series(v) for k,v in feats.items()})
# idf-weighted address overlap
qt=d.select('q_row','s1_row',t=pl.col('qa').str.split(' ')).explode('t').filter(pl.col('t')!='').unique().join(idf,on='t',how='left').with_columns(pl.col('idf').fill_null(float(np.log(1+N)+1)))
st=d.select('q_row','s1_row',t=pl.col('sa').str.split(' ')).explode('t').filter(pl.col('t')!='').unique().with_columns(ins=pl.lit(1))
qt=qt.join(st,on=['q_row','s1_row','t'],how='left')
ov=qt.group_by('q_row','s1_row').agg(q_idf=pl.col('idf').sum(), sh_idf=(pl.col('idf')*pl.col('ins').fill_null(0)).sum(), sh_max=(pl.col('idf')*pl.col('ins').fill_null(0)).max(), n_sh=pl.col('ins').fill_null(0).sum())
d=d.join(ov,on=['q_row','s1_row'],how='left').with_columns(a_idf_frac=pl.col('sh_idf')/pl.col('q_idf'))
# numbers
first=lambda c: pl.col(c).str.split(' ').list.first()
d=d.with_columns(qns=pl.col('qnum').str.split(' ').list.eval(pl.element().filter(pl.element()!='')), sns=pl.col('snum').str.split(' ').list.eval(pl.element().filter(pl.element()!='')))
d=d.with_columns(num_sh=pl.col('qns').list.set_intersection(pl.col('sns')).list.len(), num_q=pl.col('qns').list.len(), num_s=pl.col('sns').list.len(),
                 num_first_eq=(first('qnum')==first('snum')).cast(pl.Int8),
                 num_d=(first('qnum').str.slice(0,9).cast(pl.Int64,strict=False)-first('snum').str.slice(0,9).cast(pl.Int64,strict=False)))
d=d.with_columns(num_gen=pl.col('num_d').is_in([1,2,3,4,5,7,9,11,13,21]).cast(pl.Int8), num_frac=pl.col('num_sh')/pl.col('num_q').clip(1),
                 qtw=pl.lit(0), len_qn=pl.col('qn').str.len_chars(), len_sn=pl.col('sn').str.len_chars(), ntok_qa=pl.col('qa').str.count_matches(' ')+1, ntok_sa=pl.col('sa').str.count_matches(' ')+1)
F=['src','blk_score','blk_rank','p5','n_tset','n_tsort','n_part','n_r','sq_jw','a_tset','a_part','q_idf','sh_idf','sh_max','n_sh','a_idf_frac','num_sh','num_q','num_s','num_first_eq','num_d','num_gen','num_frac','tw','len_qn','len_sn','ntok_qa','ntok_sa']
out=d.select('q_row','s1_row',*F)
if split=='train':
    gt=io.gt_rows().with_columns(y=pl.lit(1,pl.Int8))
    out=out.join(gt,on=['q_row','s1_row'],how='left').with_columns(pl.col('y').fill_null(0))
    print('positives',out['y'].sum(),f'of {out.height:,}')
out.write_parquet(f'{R}/rf_{part}_{C}.parquet')
print('done',f'({time.time()-t0:.0f}s)')
