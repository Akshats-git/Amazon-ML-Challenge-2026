"""Rescue v4: v2 features + rescue cross-encoder logit (rce, cross-fitted) and its within-query rank/gap/margin.
usage: r20_rescue4.py COUNTRY"""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os, sys, json, numpy as np, polars as pl, lightgbm as lgb
os.chdir(ROOT)
from ber import io
from ber.metrics import f05, entity_counts
from ber.pools import load_partition
S=SP; C=sys.argv[1]
F0=['src','blk_score','blk_rank','p5','n_tset','n_tsort','n_part','n_r','sq_jw','a_tset','a_part','q_idf','sh_idf','sh_max','n_sh','a_idf_frac','num_sh','num_q','num_s','num_first_eq','num_d','num_gen','num_frac','tw','len_qn','len_sn','ntok_qa','ntok_sa']
KEYS=['a_idf_frac','sh_idf','a_tset','a_part','n_tset','n_r','sq_jw','blk_score','num_frac','num_sh']
def qknown(part):
    split='test' if part=='test' else 'train'
    s1_rows,q_rows=load_partition(part,C)
    s1=io.load_norm(split,'s1',columns=['name_norm']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32)).filter(pl.col('s1_row').is_in(pl.Series(s1_rows).implode()))
    voc=s1.select(t=pl.col('name_norm').str.split(' ')).explode('t').unique()
    q=io.load_norm(split,'q',columns=['name_norm']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32)).filter(pl.col('q_row').is_in(pl.Series(q_rows).implode()))
    qt=q.select('q_row',t=pl.col('name_norm').str.split(' ')).explode('t').filter(pl.col('t')!='')
    qt=qt.join(voc.with_columns(k=pl.lit(1)),on='t',how='left')
    return qt.group_by('q_row').agg(q_known=pl.col('k').fill_null(0).mean().cast(pl.Float32), q_ntok=pl.len().cast(pl.Int16))
def rel(d):
    ex=[pl.len().over('q_row').alias('n_c'),
        (pl.col('num_first_eq')==1).sum().over('q_row').alias('n_numeq'),
        (pl.col('a_tset')>=90).sum().over('q_row').alias('n_a90'),
        (pl.col('n_tset')>=80).sum().over('q_row').alias('n_n80')]
    for k in KEYS:
        ex+= [pl.col(k).rank('min',descending=True).over('q_row').cast(pl.Float32).alias(f'rk_{k}'),
              (pl.col(k).max().over('q_row')-pl.col(k)).cast(pl.Float32).alias(f'gap_{k}')]
    d=d.with_columns(ex)
    # margin over the best OTHER candidate (second best if I am the top)
    for k in ('a_idf_frac','sh_idf','n_tset','sq_jw'):
        d=d.with_columns(pl.col(k).top_k(2).over('q_row',mapping_strategy='join').alias('_t2'))
        d=d.with_columns((pl.col(k)-pl.when(pl.col(k)>=pl.col('_t2').list.first()).then(pl.col('_t2').list.get(-1,null_on_oob=True)).otherwise(pl.col('_t2').list.first())).cast(pl.Float32).alias(f'mg_{k}')).drop('_t2')
    return d
CE=os.environ.get('RCE',f'{R}/rce/out')
def cefeat(p):
    if p=='test':
        z=[pl.read_parquet(f'{CE}/rxl_{k}_test_{C}.parquet') for k in ('P0','P1')]
        c=z[0].join(z[1].rename({'ce':'ce2'}),on=['q_row','s1_row']).select('q_row','s1_row',ce=((pl.col('ce')+pl.col('ce2'))/2).cast(pl.Float32))
    else:
        o='P1' if p=='P0' else 'P0'
        c=pl.read_parquet(f'{CE}/rxl_{o}_{p}_{C}.parquet').select('q_row','s1_row',pl.col('ce').cast(pl.Float32))
    return c
d={}
for p in ('P0','P1','test'):
    x=pl.read_parquet(f'{R}/rf_{p}_{C}.parquet').join(cefeat(p),on=['q_row','s1_row'],how='left')
    x=x.with_columns(ce_rk=pl.col('ce').rank('min',descending=True).over('q_row').cast(pl.Float32),
                     ce_gap=(pl.col('ce').max().over('q_row')-pl.col('ce')).cast(pl.Float32),
                     ce_n=pl.col('ce').is_not_null().sum().over('q_row').cast(pl.Float32))
    x=x.with_columns(pl.col('ce').top_k(2).over('q_row',mapping_strategy='join').alias('_t2'))
    x=x.with_columns(ce_mg=(pl.col('ce')-pl.when(pl.col('ce')>=pl.col('_t2').list.first()).then(pl.col('_t2').list.get(-1,null_on_oob=True)).otherwise(pl.col('_t2').list.first())).cast(pl.Float32)).drop('_t2')
    d[p]=x; print(p,x.height,'ce non-null',x['ce'].is_not_null().sum(),flush=True)
F=F0+['ce','ce_rk','ce_gap','ce_n','ce_mg']
P={'objective':'binary','learning_rate':0.05,'num_leaves':127,'min_data_in_leaf':100,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':1,'lambda_l2':1.0,'verbose':-1,'num_threads':32}
if os.environ.get('NOCE')=='1': F=F0
models={}
for tr,te in (('P0','P1'),('P1','P0')):
    x=d[tr]; qs=x['q_row'].unique().sample(fraction=0.1,seed=1)
    hold=x['q_row'].is_in(qs.implode())
    xt,xh=x.filter(~hold),x.filter(hold)
    dt=lgb.Dataset(xt.select(F).to_numpy().astype(np.float32),xt['y'].to_numpy()); dh=lgb.Dataset(xh.select(F).to_numpy().astype(np.float32),xh['y'].to_numpy(),reference=dt)
    m=lgb.train(P,dt,3000,valid_sets=[dh],callbacks=[lgb.early_stopping(100,verbose=False)])
    models[tr]=m
    d[te]=d[te].with_columns(r=pl.Series(m.predict(d[te].select(F).to_numpy().astype(np.float32),num_threads=32).astype(np.float32)))
    print(f'model {tr}: best_iter {m.best_iteration} holdout logloss {m.best_score["valid_0"]["binary_logloss"]:.5f}',flush=True)
imp=sorted(zip(F,models['P0'].feature_importance('gain')),key=lambda z:-z[1]); tot=sum(v for _,v in imp); print('top gain',[(f,round(g/tot,3)) for f,g in imp[:12]])
th=json.load(open(f'{S}/p8b_thresholds.json')); T1,T2=th['T1'],th['T2']
isin=1 if C=='India' else 0
r=pl.read_parquet(f'{S}/p8b_oof_rows.parquet').filter(pl.col('is_india')==isin).sort(['q_row','p5'],descending=[False,True])
top=r.group_by('q_row',maintain_order=True).agg(pl.col('s1_row').first(),p=pl.col('p5').first(),second=pl.col('p5').slice(1,1).first())
top=top.filter(pl.col('second').is_null()|((pl.col('p')-pl.col('second'))>=1e-6)).with_columns(rk=pl.col('p').rank('ordinal',descending=True).over('s1_row'))
top=top.with_columns(link=pl.when(pl.col('rk')==1).then(pl.col('p')>=T1).otherwise(pl.col('p')>=T2))
pred=top.filter('link').select('s1_row','q_row')
gt=io.gt_rows()
uni=np.concatenate([np.load(f'work/pools/{p}_{C}.npz')['eval_s1'] for p in ('P0','P1')])
qa=np.concatenate([np.load(f'work/pools/{p}_{C}.npz')['q_rows'] for p in ('P0','P1')])
truth=gt.filter(pl.col('s1_row').is_in(pl.Series(uni).implode())&pl.col('q_row').is_in(pl.Series(qa).implode()))
ts,tq=truth['s1_row'].to_numpy(),truth['q_row'].to_numpy()
def Fm(pr): return f05(*entity_counts(uni,pr['s1_row'].to_numpy(),pr['q_row'].to_numpy(),ts,tq)).mean()
f0=Fm(pred); print(f'{C} base p8b OOF F {f0:.5f} links {pred.height:,}',flush=True)
allr=pl.concat([d['P0'].select('q_row','s1_row','r','y'),d['P1'].select('q_row','s1_row','r','y')]).sort(['q_row','r'],descending=[False,True]).group_by('q_row',maintain_order=True).agg(pl.col('s1_row').first(),r=pl.col('r').first(),y=pl.col('y').first())
allr=allr.join(pred.select('q_row'),on='q_row',how='anti')
n_s2=pl.scan_parquet(io.norm_path('train',2)).select(pl.len()).collect().item()
best=(0,None)
for tau in (0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95):
    add=allr.filter(pl.col('r')>=tau).select('s1_row','q_row')
    pr=pl.concat([pred,add]).unique()
    pr=pr.with_columns(s3=pl.col('q_row')>=n_s2).with_columns(k=pl.int_range(pl.len()).over('s1_row','s3'))
    pr=pr.filter(pl.col('k')<pl.when(pl.col('s3')).then(6).otherwise(5)).select('s1_row','q_row')
    f=Fm(pr); a=allr.filter(pl.col('r')>=tau)
    print(f'  tau {tau}: add {a.height:,} (true {a["y"].sum():,}, prec {a["y"].mean() if a.height else 0:.3f})  F {f:.5f}  delta {f-f0:+.5f}',flush=True)
    if f-f0>best[0]: best=(f-f0,tau)
print('best',best)
json.dump({'tau':best[1],'delta':best[0]},open(f'{R}/rescue4_{C}.json','w'))
X=d['test'].select(F).to_numpy().astype(np.float32)
rt=np.mean([m.predict(X,num_threads=32) for m in models.values()],axis=0).astype(np.float32)
bt=d['test'].select('q_row','s1_row').with_columns(r=pl.Series(rt)).sort(['q_row','r'],descending=[False,True]).group_by('q_row',maintain_order=True).agg(pl.col('s1_row').first(),r=pl.col('r').first())
bt.write_parquet(f'{R}/rescue4_best_test_{C}.parquet')
add=bt.filter(pl.col('r')>=best[1]).select('s1_row','q_row')
print(f'{C} test additions at tau {best[1]}: {add.height:,}')
add.write_parquet(f'{R}/rescue4_add_test_{C}.parquet')
