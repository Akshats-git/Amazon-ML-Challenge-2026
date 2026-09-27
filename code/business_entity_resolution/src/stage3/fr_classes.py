import os as _os, sys as _sys
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
_os.makedirs(SP, exist_ok=True)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import polars as pl
from rapidfuzz.distance import Levenshtein
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)

GEN=[1,2,3,4,5,7,9,11,13,21]
LEG = {'sarl','sas','sasu','sa','eurl','sci','snc','scop','selarl','sca','gie','scs','scm','earl','gaec'}
TN = {'groupe','developement','france','cie','services','fils','asocies','and','et','compagnie'}
PURE = {'participations','holding','international','distribution'}
STOPF = {'de','du','des','la','le','les','l','d','a','au','aux','en'}
def join1(s):
    out, buf = [], []
    for w in s.split() + ['']:
        if len(w) == 1: buf.append(w); continue
        if len(buf) >= 2: out.append(''.join(buf))
        elif buf: out.extend(buf)
        buf = []
        if w: out.append(w)
    return [w for w in out if w not in LEG and w not in STOPF]
t = pl.read_parquet(f'{SP}/g12_test_France.parquet', columns=['q_row','s1_row','p','name_norm','name_norm_s','addr_nums','addr_nums_s'])
first = lambda x: pl.col(x).str.split(' ').list.first().str.slice(0,9).cast(pl.Int64, strict=False)
t = t.with_columns(d=first('addr_nums')-first('addr_nums_s'))
t = t.with_columns(num=pl.when(pl.col('addr_nums')=='').then(pl.lit('na')).when(pl.col('addr_nums')==pl.col('addr_nums_s')).then(pl.lit('eq')).when(pl.col('d').is_in(GEN)).then(pl.lit('gen')).otherwise(pl.lit('other')))
t = t.with_columns(qc=pl.col('name_norm').map_elements(join1, return_dtype=pl.List(pl.Utf8)), sc=pl.col('name_norm_s').map_elements(join1, return_dtype=pl.List(pl.Utf8)))
t = t.with_columns(E=pl.col('qc').list.set_difference(pl.col('sc')), M=pl.col('sc').list.set_difference(pl.col('qc')))
# category vocabulary: words swapped in by Type A distractors (+GEN single swaps), minus TN / pure / legal
g1 = t.filter((pl.col('num')=='gen') & (pl.col('E').list.len()==1) & (pl.col('M').list.len()==1)).select(w=pl.col('E').list.first()).group_by('w').len()
CAT = set(g1.filter(pl.col('len') >= 300)['w'].to_list()) - TN - PURE - LEG
print(len(CAT), sorted(CAT)[:80])
cat = list(CAT)
def typo(E, M):
    if not E or not M: return False
    return all(max(Levenshtein.normalized_similarity(e, m) for m in M) >= 0.7 for e in E)
t = t.with_columns(typo=pl.struct('E','M').map_elements(lambda r: typo(r['E'], r['M']), return_dtype=pl.Boolean))
acr = pl.struct('qc','sc').map_elements(lambda r: len(r['qc'])==1 and 2 <= len(r['qc'][0]) <= 6 and len(r['sc']) >= 2 and r['qc'][0] == ''.join(w[0] for w in r['sc']), return_dtype=pl.Boolean)
t = t.with_columns(acr=acr)
E_, M_ = pl.col('E'), pl.col('M')
t = t.with_columns(cls=pl.when(E_.list.len()+M_.list.len()==0).then(pl.lit('same_core'))
    .when(pl.col('acr')).then(pl.lit('acronym'))
    .when(pl.col('typo')).then(pl.lit('typo'))
    .when(E_.list.eval(pl.element().is_in(list(TN))).list.all() & (E_.list.len()>0)).then(pl.lit('TN_in'))
    .when(E_.list.eval(pl.element().is_in(list(PURE))).list.any()).then(pl.lit('pure_gen_word'))
    .when(E_.list.eval(pl.element().is_in(cat)).list.any() & M_.list.eval(pl.element().is_in(cat)).list.any()).then(pl.lit('cat_swap'))
    .when(E_.list.len()==0).then(pl.lit('drop_only'))
    .when(pl.col('qc').list.set_intersection(pl.col('sc')).list.len()==0).then(pl.lit('no_shared'))
    .when(E_.list.eval(pl.element().is_in(cat)).list.any()).then(pl.lit('cat_added'))
    .otherwise(pl.lit('rest')))
for num in ('eq','gen'):
    g = t.filter(pl.col('num')==num).group_by('cls').agg(n=pl.len(), linked=(pl.col('p')>=0.52).sum(), unc=((pl.col('p')>0.02)&(pl.col('p')<0.98)).sum(), mp=pl.col('p').mean()).sort('n', descending=True)
    print('== num', num); print(g)
t.select('q_row','s1_row','num','cls').write_parquet(f'{SP}/fr_cls.parquet')
print(t.filter((pl.col('num')=='eq') & (pl.col('cls')=='rest')).sample(20, seed=4).select('p','name_norm','name_norm_s'))
