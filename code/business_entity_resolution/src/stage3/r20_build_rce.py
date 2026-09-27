"""Rescue CE sets: score rf_{part}_{C} with the v2 rescue models (cross-fit for pools), keep queries with max r >= RMIN,
top-TOPK candidates per query, attach raw 'name | address' text. usage: r20_build_rce.py C"""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os, sys, numpy as np, polars as pl, lightgbm as lgb
os.chdir(ROOT)
from ber import io
os.makedirs(f'{R}/rce',exist_ok=True)
C=sys.argv[1]; RMIN=float(os.environ.get('RMIN','0.03')); TOPK=int(os.environ.get('TOPK','4'))
F=['src','blk_score','blk_rank','p5','n_tset','n_tsort','n_part','n_r','sq_jw','a_tset','a_part','q_idf','sh_idf','sh_max','n_sh','a_idf_frac','num_sh','num_q','num_s','num_first_eq','num_d','num_gen','num_frac','tw','len_qn','len_sn','ntok_qa','ntok_sa']
kw=dict(separator='\t',quote_char=None,infer_schema=False)
txt=lambda df: df.with_columns(t=pl.col('business_name').fill_null('')+' | '+pl.col('business_address').fill_null('')).select('entity_id','t')
for part in ('P0','P1','test'):
    split='test' if part=='test' else 'train'
    d=pl.read_parquet(f'{R}/rf_{part}_{C}.parquet')
    X=d.select(F).to_numpy().astype(np.float32)
    mods=['P1'] if part=='P0' else ['P0'] if part=='P1' else ['P0','P1']
    r=np.mean([lgb.Booster(model_file=f'{R}/rescue_{C}_{k}.txt').predict(X,num_threads=32) for k in mods],axis=0).astype(np.float32)
    d=d.with_columns(r2=pl.Series(r)).drop([c for c in F if c not in ('p5',)])
    d=d.with_columns(mx=pl.col('r2').max().over('q_row'),rk=pl.col('r2').rank('ordinal',descending=True).over('q_row'))
    d=d.filter((pl.col('mx')>=RMIN)&(pl.col('rk')<=TOPK))
    q=io.load_norm(split,'q',columns=['entity_id']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32))
    s1=pl.read_parquet(f'work/norm/{split}_s1.parquet',columns=['entity_id']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
    d=d.join(q.rename({'entity_id':'qid'}),on='q_row').join(s1.rename({'entity_id':'sid'}),on='s1_row')
    rq=pl.concat([pl.scan_csv(f'data/raw/{split}/{split}_source{s}.tsv',**kw).filter(pl.col('entity_id').is_in(d['qid'].implode())).collect() for s in (2,3)])
    r1=pl.scan_csv(f'data/raw/{split}/{split}_source1.tsv',**kw).filter(pl.col('entity_id').is_in(d['sid'].implode())).collect()
    d=d.join(txt(rq).rename({'entity_id':'qid','t':'a'}),on='qid',how='left').join(txt(r1).rename({'entity_id':'sid','t':'b'}),on='sid',how='left')
    d=d.with_columns(pl.col('a').fill_null(''),pl.col('b').fill_null(''))
    if 'y' in d.columns: d=d.with_columns(ok=pl.col('y'))
    d.write_parquet(f'{R}/rce/rce_{part}_{C}.parquet')
    print(part,C,'queries',d['q_row'].n_unique(),'pairs',d.height,('pos %d'%d['y'].sum()) if 'y' in d.columns else '',flush=True)
