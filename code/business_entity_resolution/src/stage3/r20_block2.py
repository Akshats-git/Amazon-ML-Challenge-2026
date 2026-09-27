import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, sys, time, numpy as np, polars as pl
os.chdir(ROOT); pass
from ber import io, blocking as B, config as C
from ber.pools import load_partition
C.BLOCK_THREADS=16
part, cty = sys.argv[1], sys.argv[2]
W = tuple(float(x) for x in sys.argv[3].split(','))
K = int(sys.argv[4]) if len(sys.argv)>4 else 10
tag = sys.argv[5] if len(sys.argv)>5 else 'a'
C.BLOCK_WEIGHTS = W
split = 'test' if part=='test' else 'train'
s1n = io.load_norm(split,'s1',columns=B.BLOCK_COLS); qn = io.load_norm(split,'q',columns=B.BLOCK_COLS)
s1_rows, q_rows = load_partition(part, cty)
t=time.time()
c2 = B.block_one(s1n[s1_rows], qn[q_rows], s1_rows, q_rows, k=K)
import glob
old = pl.read_parquet(B.cand_path(part, cty), columns=['q_row','s1_row']) if B.cand_path(part, cty).exists() else pl.concat([pl.read_parquet(f, columns=['q_row','s1_row']) for f in sorted(glob.glob(f'work/feats/{part}/{cty}_part-*.parquet'))])
new = c2.join(old, on=['q_row','s1_row'], how='anti')
print(f'{part} {cty} W={W} K={K}: new pairs {new.height:,} of {c2.height:,}  ({time.time()-t:.0f}s)')
new.write_parquet(fR+'/b2_{tag}_{part}_{cty}.parquet')
if split=='train':
    gt = io.gt_rows()
    truth = gt.filter(pl.col('s1_row').is_in(pl.Series(s1_rows).implode()) & pl.col('q_row').is_in(pl.Series(q_rows).implode()))
    bo = truth.join(old, on=['q_row','s1_row'], how='anti')
    rec = bo.join(new, on=['q_row','s1_row'], how='semi')
    print(f'  truth {truth.height:,}  blocked-out {bo.height:,}  recovered by new {rec.height:,} ({rec.height/bo.height:.3f})')
    qe = io.load_norm(split,'q',columns=['addr_norm']).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32))
    x = bo.join(qe, on='q_row').with_columns(e=pl.col('addr_norm')=='').join(new.select('q_row','s1_row',pl.lit(1).alias('r')),on=['q_row','s1_row'],how='left')
    print(x.group_by('e').agg(n=pl.len(), rec=pl.col('r').sum()))
