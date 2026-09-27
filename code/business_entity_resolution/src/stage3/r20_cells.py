import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import polars as pl, sys
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)

from r20_lib import prep
LEG={'US':{'llc','inc','corp','co','ltd','lp','llp','pllc','pc','company','corporation','incorporated','limited','lc'},
     'India':{'pvt','ltd','private','limited','llp','opc','lp','public'},
     'France':{'sarl','sas','sasu','sa','eurl','sci','snc','scop','selarl','sca','gie','scs','scm','earl','gaec','ei'}}
STOP={'de','du','des','la','le','les','d','l','au','aux','en','of','and','the','et','a'}
def cells(t, C):
    lg=list(LEG[C]); st=list(STOP)
    tok=lambda c: pl.col(c).str.split(' ').list.eval(pl.element().filter(pl.element()!=''))
    # join runs of single letters (s a r l -> sarl) is ignored; single letters treated as stop
    core=lambda c: tok(c).list.eval(pl.element().filter(~pl.element().is_in(lg+st) & (pl.element().str.len_chars()>1)))
    legal=lambda c: tok(c).list.eval(pl.element().filter(pl.element().is_in(lg))).list.sort()
    t=t.with_columns(qc=core('name_norm').list.sort(), sc=core('name_norm_s').list.sort(), ql=legal('name_norm'), sl=legal('name_norm_s'))
    vocab=set(t['name_norm_s'].str.split(' ').explode().value_counts().filter(pl.col('count')>=5)['name_norm_s'].to_list())
    t=t.with_columns(E=pl.col('qc').list.set_difference(pl.col('sc')), M=pl.col('sc').list.set_difference(pl.col('qc')),
                     I=pl.col('qc').list.set_intersection(pl.col('sc')).list.len(), nq=pl.col('qc').list.len(), ns=pl.col('sc').list.len())
    t=t.with_columns(nE=pl.col('E').list.len(), nM=pl.col('M').list.len(),
                     oovE=pl.col('E').list.eval(~pl.element().is_in(list(vocab))).list.all(),
                     acr=(pl.col('nq')==1)&(pl.col('ns')>=2)&(pl.col('qc').list.first()==pl.col('sc').list.eval(pl.element().str.slice(0,1)).list.join('')),
                     cat=(pl.col('nq')==1)&(pl.col('ns')>=2)&(pl.col('I')==0))
    t=t.with_columns(nrel=pl.when((pl.col('qc')==pl.col('sc'))).then(pl.lit('same'))
        .when(pl.col('nq')==0).then(pl.lit('q_empty_core'))
        .when((pl.col('I')==0)&(pl.col('nq')==1)&pl.col('oovE')).then(pl.lit('noshare_1oov'))
        .when((pl.col('I')==0)&(pl.col('nq')==1)).then(pl.lit('noshare_1iv'))
        .when((pl.col('I')==0)&pl.col('oovE')).then(pl.lit('noshare_Moov'))
        .when(pl.col('I')==0).then(pl.lit('noshare_Miv'))
        .when((pl.col('nM')==0)&(pl.col('nE')==1)&pl.col('oovE')).then(pl.lit('add1_oov'))
        .when((pl.col('nM')==0)&(pl.col('nE')==1)).then(pl.lit('add1_iv'))
        .when(pl.col('nM')==0).then(pl.lit('add2+'))
        .when((pl.col('nE')==0)&(pl.col('nM')==1)).then(pl.lit('drop1'))
        .when(pl.col('nE')==0).then(pl.lit('drop2+'))
        .when((pl.col('nE')==1)&(pl.col('nM')==1)&pl.col('oovE')).then(pl.lit('swap1_oov'))
        .when((pl.col('nE')==1)&(pl.col('nM')==1)).then(pl.lit('swap1_iv'))
        .otherwise(pl.lit('multi')),
        lrel=pl.when(pl.col('ql')==pl.col('sl')).then(pl.lit('Lsame')).when(pl.col('ql').list.len()==0).then(pl.lit('Ldrop')).when(pl.col('sl').list.len()==0).then(pl.lit('Ladd')).otherwise(pl.lit('Lchg')),
        st=pl.when(pl.col('jac')>=0.6).then(pl.lit('S')).otherwise(pl.lit('s')))
    hh=pl.when(pl.col('h').is_in(['eq','gen','na'])).then(pl.col('h')).when(pl.col('d').is_in([-1,-2])).then(pl.lit('neg12')).when(pl.col('h')=='s_na').then(pl.lit('s_na')).otherwise(pl.lit('other'))
    return t.with_columns(cell=pl.concat_str([hh, pl.col('st'), pl.col('nrel'), pl.col('lrel')], separator='|'))
cols=['q_row','s1_row','p','name_norm','addr_norm','addr_nums','name_norm_s','addr_norm_s','addr_nums_s']
res=[]
for C in ('US','India'):
    t=cells(prep(pl.read_parquet(f'{SP}/g_P0_{C}.parquet',columns=cols+['ok'])),C)
    res.append(t.group_by('cell').agg(pl.len().alias(f'n_{C}'), pl.col('ok').mean().alias(f't_{C}')))
f=cells(prep(pl.read_parquet(R+'/frg.parquet')),'France')
f.select('q_row','s1_row','cell','L','p','xl').write_parquet(R+'/fr_cells.parquet')
fa=f.group_by('cell').agg(n_F=pl.len(), L_F=pl.col('L').mean(), U_F=(1-pl.col('L')).sum(), Lk_F=pl.col('L').sum(), p_F=pl.col('p').mean(), xl_F=pl.col('xl').mean())
m=res[0].join(res[1],on='cell',how='full',coalesce=True).join(fa,on='cell',how='full',coalesce=True).fill_null(0)
m=m.with_columns(tUI=(pl.col('t_US')*pl.col('n_US')+pl.col('t_India')*pl.col('n_India'))/(pl.col('n_US')+pl.col('n_India')), agree=(pl.col('t_US')-pl.col('t_India')).abs())
m.write_parquet(R+'/cells_table.parquet')
ok=m.filter((pl.col('n_US')>=200)&(pl.col('n_India')>=200)&(pl.col('agree')<=0.06)&(pl.col('n_F')>=150))
# expected France gain if decisions follow tUI: add unlinked where tUI>0.72, remove linked where tUI<0.72
ok=ok.with_columns(dev=pl.col('L_F')-pl.col('tUI'))
ok=ok.with_columns(gain_add=pl.when(pl.col('tUI')>0.75).then(pl.col('U_F')*(pl.col('tUI')*0.09-(1-pl.col('tUI'))*0.21)).otherwise(0),
                   gain_rem=pl.when(pl.col('tUI')<0.65).then(pl.col('Lk_F')*((1-pl.col('tUI'))*0.21-pl.col('tUI')*0.09)).otherwise(0))
print(ok.filter(pl.col('dev').abs()>=0.04).sort(pl.col('gain_add')+pl.col('gain_rem'),descending=True).select('cell','n_US','t_US','n_India','t_India','n_F','L_F','U_F','Lk_F','p_F','xl_F','gain_add','gain_rem').with_columns(pl.col(pl.Float64).round(3)).head(60))
