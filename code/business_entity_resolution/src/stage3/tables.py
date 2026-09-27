"""Argmax tables for stage 3: one row per query (argmax of the x3+x4 blend), with p of every model at that pair, the
runner-up p, normalized text of both sides and raw-text perturbation counts. part in {P0, P1, test}."""
import os as _os, sys as _sys
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
_os.makedirs(SP, exist_ok=True)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, os, sys
import numpy as np, polars as pl
os.chdir(ROOT)
W = 'work'; SP = SP
part, c = sys.argv[1], sys.argv[2]
split = 'test' if part == 'test' else 'train'
files = sorted(glob.glob(f'{W}/feats/{part}/{c}_part-*.parquet'))
k = pl.concat([pl.read_parquet(f, columns=['q_row', 's1_row'] + (['y'] if split == 'train' else [])) for f in files])
P = {t: np.load(f'{W}/oof/{t}/p2_{part}_{c}.npy') for t in ('x1', 'x2', 'x3', 'x4')}
for t in P:
    assert len(P[t]) == k.height, t
BL = os.environ.get('BLEND', 'x3,x4').split(','); p = (sum(P[b] for b in BL) / len(BL)).astype(np.float32)
q_row = k['q_row'].to_numpy(); s1_row = k['s1_row'].to_numpy()
new = np.r_[True, q_row[1:] != q_row[:-1]]
starts = np.flatnonzero(new); gid = np.cumsum(new) - 1
idx = np.flatnonzero(p == np.maximum.reduceat(p, starts)[gid])
am = idx[np.r_[True, gid[idx][1:] != gid[idx][:-1]]]
rest = p.copy(); rest[am] = -np.inf
d = {'q_row': q_row[starts], 's1_row': s1_row[am], 'p': p[am], 'p2nd': np.maximum.reduceat(rest, starts),
     'n_cand': np.diff(np.r_[starts, len(p)]).astype(np.int16)}
for t in P:
    d[f'p_{t}'] = P[t][am].astype(np.float32)
if split == 'train':
    y = k['y'].to_numpy()
    d['ok'] = y[am] == 1
    d['has_true'] = np.maximum.reduceat(y, starts) == 1
top = pl.DataFrame(d)
del k, P, p, rest, q_row, s1_row
cols = ['name_norm', 'addr_norm', 'addr_nums']
s1 = pl.read_parquet(f'{W}/norm/{split}_s1.parquet', columns=cols + ['entity_id', 'country']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32)).filter(pl.col('country') == c).drop('country')
q = pl.concat([pl.read_parquet(f'{W}/norm/{split}_s{s}.parquet', columns=cols + ['entity_id', 'country']) for s in (2, 3)]).with_row_index('q_row').with_columns(pl.col('q_row').cast(pl.Int32)).filter(pl.col('country') == c).drop('country')
top = top.join(q, on='q_row').join(s1.rename({x: x + '_s' for x in cols + ['entity_id']}), on='s1_row')
del q, s1
# raw text of both sides
R = f'data/raw/{split}'
kw = dict(separator='\t', quote_char=None, infer_schema=False)
rawq = pl.concat([pl.scan_csv(f'{R}/{split}_source{s}.tsv', **kw).filter(pl.col('country') == c).select('entity_id', 'business_name', 'business_address').collect() for s in (2, 3)])
raw1 = pl.scan_csv(f'{R}/{split}_source1.tsv', **kw).filter(pl.col('country') == c).select('entity_id', 'business_name', 'business_address').collect()
top = top.join(rawq.rename({'business_name': 'qn', 'business_address': 'qa'}), on='entity_id', how='left')
top = top.join(raw1.rename({'entity_id': 'entity_id_s', 'business_name': 'sn', 'business_address': 'sa'}), on='entity_id_s', how='left')
del rawq, raw1
top = top.with_columns([pl.col(x).fill_null('') for x in ['qn', 'qa', 'sn', 'sa']])
dbl = '|'.join(ch * 2 for ch in 'abcdefghijklmnopqrstuvwxyz')
tri = '|'.join(ch * 3 for ch in 'abcdefghijklmnopqrstuvwxyz')
L = lambda x: pl.col(x).str.to_lowercase()
rep = lambda x: pl.col(x).str.to_lowercase().str.extract_all(r'[a-z0-9]+').map_elements(lambda w: sum(1 for a, b in zip(w, w[1:]) if a == b), return_dtype=pl.Int32)
top = top.with_columns(
    q_dbl=L('qn').str.count_matches(dbl).cast(pl.Int32), s_dbl=L('sn').str.count_matches(dbl).cast(pl.Int32),
    q_tri=L('qn').str.count_matches(tri).cast(pl.Int32),
    q_acc=pl.col('qn').str.count_matches(r'[À-ÿ]').cast(pl.Int32), s_acc=pl.col('sn').str.count_matches(r'[À-ÿ]').cast(pl.Int32),
    q_numsfx=pl.col('qa').str.count_matches(r'(?i)\b\d+-?[a-z]\b').cast(pl.Int32), s_numsfx=pl.col('sa').str.count_matches(r'(?i)\b\d+-?[a-z]\b').cast(pl.Int32),
    q_frac=pl.col('qa').str.count_matches(r'\b1/2\b').cast(pl.Int32), s_frac=pl.col('sa').str.count_matches(r'\b1/2\b').cast(pl.Int32),
    q_brack=pl.col('qn').str.count_matches(r'\[').cast(pl.Int32), q_paren=pl.col('qn').str.count_matches(r'\(').cast(pl.Int32),
    q_hash=pl.col('qa').str.count_matches(r'#\d').cast(pl.Int32),
    q_rep=rep('qn'), s_rep=rep('sn'),
).drop(['qn', 'qa', 'sn', 'sa', 'entity_id_s'])
top.write_parquet(f"{SP}/{os.environ.get('PFX', 'g')}_{part}_{c}.parquet")
print(part, c, top.height, 'rows', ('ok %.4f' % top['ok'].mean()) if split == 'train' else '')
