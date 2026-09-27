import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import polars as pl
from rapidfuzz.distance import Levenshtein as L
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(48)

from r20_lib import prep
def pat(t):
    # first token modified, rest identical (token lists), both first tokens short
    t=t.with_columns(qt=pl.col('name_norm').str.split(' '), st_=pl.col('name_norm_s').str.split(' '))
    t=t.with_columns(q0=pl.col('qt').list.first(), s0=pl.col('st_').list.first(), qr=pl.col('qt').list.slice(1), sr=pl.col('st_').list.slice(1))
    t=t.filter((pl.col('qr')==pl.col('sr'))&(pl.col('q0')!=pl.col('s0'))&(pl.col('qt').list.len()>=2))
    t=t.with_columns(ed=pl.struct('q0','s0').map_elements(lambda r: L.distance(r['q0'],r['s0']),return_dtype=pl.Int32),
                     short=(pl.col('s0').str.len_chars()<=4))
    return t
for C in ('US','India'):
    t=prep(pl.read_parquet(f'{SP}/g_P0_{C}.parquet',columns=['q_row','s1_row','p','name_norm','addr_norm','addr_nums','name_norm_s','addr_norm_s','addr_nums_s','ok']))
    t=pat(t.filter((pl.col('h')=='eq')&(pl.col('jac')>=0.6)))
    print(C); print(t.group_by('short',pl.col('ed').clip(1,3)).agg(n=pl.len(),true=pl.col('ok').mean(),p=pl.col('p').mean()).sort('short','ed'))
    print(t.filter(pl.col('short')&(pl.col('ed')==1)).sample(8,seed=1).select('ok','p','name_norm','name_norm_s'))
f=prep(pl.read_parquet(R+'/frg.parquet'))
f=pat(f.filter((pl.col('h')=='eq')&(pl.col('jac')>=0.6)))
print('France'); print(f.group_by('short',pl.col('ed').clip(1,3)).agg(n=pl.len(),L=pl.col('L').mean(),Lk=pl.col('L').sum(),p=pl.col('p').mean(),xl=pl.col('xl').mean(),xl_L=pl.col('xl').filter(pl.col('L')==1).mean()).sort('short','ed'))
print(f.filter(pl.col('short')&(pl.col('ed')==1)&(pl.col('L')==1)).sample(15,seed=1).select('p','xl','name_norm','name_norm_s'))
f.select('q_row','s1_row','L','p','xl','short','ed').write_parquet(R+'/fr_initl.parquet')
