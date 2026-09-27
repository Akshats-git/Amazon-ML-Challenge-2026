"""Score test rescue candidates with both fold models, pick best new candidate per target query, write additions."""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os, sys, json, numpy as np, polars as pl, lightgbm as lgb
C=sys.argv[1] if len(sys.argv)>1 else 'India'
tau=float(sys.argv[2]) if len(sys.argv)>2 else json.load(open(f'{R}/rescue_{C}.json'))['tau']
F=['src','blk_score','blk_rank','p5','n_tset','n_tsort','n_part','n_r','sq_jw','a_tset','a_part','q_idf','sh_idf','sh_max','n_sh','a_idf_frac','num_sh','num_q','num_s','num_first_eq','num_d','num_gen','num_frac','tw','len_qn','len_sn','ntok_qa','ntok_sa']
d=pl.read_parquet(f'{R}/rf_test_{C}.parquet')
X=d.select(F).to_numpy().astype(np.float32)
r=np.mean([lgb.Booster(model_file=f'{R}/rescue_{C}_{k}.txt').predict(X,num_threads=16) for k in ('P0','P1')],axis=0).astype(np.float32)
d=d.with_columns(r=pl.Series(r))
best=d.sort(['q_row','r'],descending=[False,True]).group_by('q_row',maintain_order=True).agg(pl.col('s1_row').first(),r=pl.col('r').first())
add=best.filter(pl.col('r')>=tau).select('s1_row','q_row')
print(f'{C} test: target queries {best.height:,}  additions at tau {tau}: {add.height:,}  r quantiles', np.quantile(best['r'].to_numpy(),[0.5,0.9,0.99]).round(3))
add.write_parquet(f'{R}/rescue_add_test_{C}.parquet')
