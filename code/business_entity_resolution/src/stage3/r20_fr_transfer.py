import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250)

m=pl.read_parquet(R+'/cells_table.parquet')
SAFE_NREL=['same','noshare_1oov','noshare_1iv','drop1','drop2+','q_empty_core']
m=m.with_columns(nrel=pl.col('cell').str.split('|').list.get(2), st=pl.col('cell').str.split('|').list.get(1), hh=pl.col('cell').str.split('|').list.get(0))
good=(pl.col('n_US')>=200)&(pl.col('n_India')>=200)&(pl.col('agree')<=0.06)&(pl.col('st')=='S')
add_cells=m.filter(good&(pl.col('t_US')>=0.93)&(pl.col('t_India')>=0.93)&pl.col('nrel').is_in(SAFE_NREL)&(pl.col('hh')!='gen'))
rem_cells=m.filter(good&(pl.col('t_US')<=0.15)&(pl.col('t_India')<=0.15))
c=pl.read_parquet(R+'/fr_cells.parquet')
A=c.filter(pl.col('cell').is_in(add_cells['cell'].implode())&(pl.col('L')==0)&(pl.col('xl')>=0.8)).join(add_cells.select('cell','tUI'),on='cell')
R=c.filter(pl.col('cell').is_in(rem_cells['cell'].implode())&(pl.col('L')==1)).join(rem_cells.select('cell','tUI'),on='cell')
ini=pl.read_parquet(R+'/fr_initl.parquet').filter(pl.col('short')&(pl.col('ed')<=2)&(pl.col('L')==1)&(pl.col('xl')<0.1))
print('initialism-edit removals', ini.height)
print('ADD by cell'); print(A.group_by('cell').agg(n=pl.len(),p=pl.col('p').mean(),xl=pl.col('xl').mean(),t=pl.col('tUI').first()).sort('n',descending=True))
print('adds',A.height,'exp units (US/India rates)',round((A['tUI']*0.09-(1-A['tUI'])*0.21).sum(),1),' rem',R.height,'+ini',ini.height)
base=pl.read_parquet(f'{SP}/fr_pairs_acde.parquet')
rem=pl.concat([R.select('s1_row','q_row'),ini.select('s1_row','q_row')]).unique()
new=pl.concat([base.join(rem,on=['s1_row','q_row'],how='anti'), A.select('s1_row','q_row')]).unique()
assert new['q_row'].n_unique()==new.height, 'query linked twice'
print('base',base.height,'new',new.height)
new.write_parquet(R+'/fr_pairs_P1.parquet')
