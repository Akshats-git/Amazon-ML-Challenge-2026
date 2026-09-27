import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(50)

g=pl.read_parquet(f'{R}/frg.parquet')
P1=pl.read_parquet(f'{R}/fr_pairs_P1.parquet').with_columns(L1=pl.lit(1))
g=g.join(P1,on=['s1_row','q_row'],how='left').with_columns(pl.col('L1').fill_null(0))
V=g.filter((pl.col('L1')==1)&(pl.col('xl')<=0.05)&pl.col('cls').is_in(['same_core','drop_only','typo','no_shared'])&pl.col('num').is_in(['eq','na','other']))
print('veto candidates',V.height); print(V.group_by('num','cls').agg(n=pl.len(),p=pl.col('p').mean(),xl=pl.col('xl').mean()).sort('n',descending=True))
print(V.sample(20,seed=9).select('p','xl','num','cls','name_norm','name_norm_s','addr_norm','addr_norm_s'))
# keep the veto only where the addresses differ (address similarity < 90 after dropping region/department words), or where
# the name itself is a different business (no shared word: concatenated different names; typo: initialism edits)
from rapidfuzz import process, fuzz
strip=lambda c: pl.col(c).str.replace_all(r'\b(nouvele aquitaine|hauts de france|pays de la loire|loire atlantique|pas de calais|gironde|nord|no|ndeg)\b','').str.replace_all(r'\b0+(\d)',r'$1').str.replace_all(r'\s+',' ').str.strip_chars()
V=V.with_columns(qa=strip('addr_norm'),sa=strip('addr_norm_s'))
V=V.with_columns(asim=pl.Series(process.cpdist(V['qa'].to_list(),V['sa'].to_list(),scorer=fuzz.token_set_ratio,workers=-1)))
V=V.filter((pl.col('asim')<90)|pl.col('cls').is_in(['no_shared','typo']))
print('veto after the address-similarity rule', V.height)
rate={'eq':0.03,'na':0.09,'other':0.10}
V=V.with_columns(t=pl.col('num').replace_strict(rate,return_dtype=pl.Float64))
print('expected units if removed (0.21 per FP removed, 0.09 per TP lost):', round(((1-V['t'])*0.21-V['t']*0.09).sum(),1))
V.select('s1_row','q_row').write_parquet(f'{R}/fr_veto.parquet')
# final France pairs = structural-transfer package minus the vetoed links
P1=pl.read_parquet(f'{R}/fr_pairs_P1.parquet')
P1.join(V.select('s1_row','q_row'),on=['s1_row','q_row'],how='anti').write_parquet(f'{R}/fr_pairs_P1V.parquet')
print('written fr_pairs_P1V.parquet')
