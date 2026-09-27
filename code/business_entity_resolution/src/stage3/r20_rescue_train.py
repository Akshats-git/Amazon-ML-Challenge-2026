"""Cross-fit the rescue scorer on P0/P1 new candidates (India), evaluate OOF F gain vs the p8b decision, score test."""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os, sys, json, numpy as np, polars as pl, lightgbm as lgb
os.chdir(ROOT); pass
from ber import io
from ber.metrics import f05, entity_counts
C=sys.argv[1] if len(sys.argv)>1 else 'India'
F=['src','blk_score','blk_rank','p5','n_tset','n_tsort','n_part','n_r','sq_jw','a_tset','a_part','q_idf','sh_idf','sh_max','n_sh','a_idf_frac','num_sh','num_q','num_s','num_first_eq','num_d','num_gen','num_frac','tw','len_qn','len_sn','ntok_qa','ntok_sa']
P={'objective':'binary','learning_rate':0.05,'num_leaves':63,'min_data_in_leaf':100,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':1,'lambda_l2':1.0,'verbose':-1,'num_threads':16}
d={p:pl.read_parquet(f'{R}/rf_{p}_{C}.parquet') for p in ('P0','P1')}
models={}
for tr,te in (('P0','P1'),('P1','P0')):
    x=d[tr]; qs=x['q_row'].unique().sample(fraction=0.1,seed=1)
    hold=x['q_row'].is_in(qs.implode()) if hasattr(qs,'implode') else x['q_row'].is_in(qs)
    xt,xh=x.filter(~hold),x.filter(hold)
    dt=lgb.Dataset(xt.select(F).to_numpy().astype(np.float32),xt['y'].to_numpy()); dh=lgb.Dataset(xh.select(F).to_numpy().astype(np.float32),xh['y'].to_numpy(),reference=dt)
    m=lgb.train(P,dt,2000,valid_sets=[dh],callbacks=[lgb.early_stopping(100,verbose=False)])
    models[tr]=m
    d[te]=d[te].with_columns(r=pl.Series(m.predict(d[te].select(F).to_numpy().astype(np.float32),num_threads=16).astype(np.float32)))
    print(f'model {tr}: best_iter {m.best_iteration} holdout logloss {m.best_score["valid_0"]["binary_logloss"]:.5f}',flush=True)
imp=sorted(zip(F,models['P0'].feature_importance('gain')),key=lambda z:-z[1]); print('top gain',[(f,round(g/sum(v for _,v in imp),3)) for f,g in imp[:10]])
# base decision (p8b) for this country, both pools
th=json.load(open(f'{SP}/p8b_thresholds.json')); T1,T2=th['T1'],th['T2']
isin=1 if C=='India' else 0
r=pl.read_parquet(f'{SP}/p8b_oof_rows.parquet').filter(pl.col('is_india')==isin).sort(['q_row','p5'],descending=[False,True])
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
allr=pl.concat([d['P0'],d['P1']]).sort(['q_row','r'],descending=[False,True]).group_by('q_row',maintain_order=True).agg(pl.col('s1_row').first(),r=pl.col('r').first(),y=pl.col('y').first())
allr=allr.join(pred.select('q_row'),on='q_row',how='anti')
n_s2=pl.scan_parquet(io.norm_path('train',2)).select(pl.len()).collect().item()
best=(0,None)
for tau in (0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.97):
    add=allr.filter(pl.col('r')>=tau).select('s1_row','q_row')
    pr=pl.concat([pred,add]).unique()
    # caps
    pr=pr.with_columns(s3=pl.col('q_row')>=n_s2).with_columns(k=pl.int_range(pl.len()).over('s1_row','s3'))
    pr=pr.filter(pl.col('k')<pl.when(pl.col('s3')).then(6).otherwise(5)).select('s1_row','q_row')
    f=Fm(pr); a=allr.filter(pl.col('r')>=tau)
    print(f'  tau {tau}: add {a.height:,} (true {a["y"].sum():,}, prec {a["y"].mean() if a.height else 0:.3f})  F {f:.5f}  delta {f-f0:+.5f}',flush=True)
    if f-f0>best[0]: best=(f-f0,tau)
print('best',best)
json.dump({'tau':best[1],'delta':best[0]},open(f'{R}/rescue_{C}.json','w'))
for k,m in models.items(): m.save_model(f'{R}/rescue_{C}_{k}.txt')
