import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import polars as pl, random
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(70)

GEN=[1,2,3,4,5,7,9,11,13,21]
def prep(t):
    first=lambda x: pl.col(x).str.split(' ').list.first()
    t=t.with_columns(a=first('addr_nums'), b=first('addr_nums_s'))
    t=t.with_columns(d=pl.col('a').str.slice(0,9).cast(pl.Int64,strict=False)-pl.col('b').str.slice(0,9).cast(pl.Int64,strict=False))
    t=t.with_columns(h=pl.when(pl.col('addr_nums')=='').then(pl.lit('na')).when(pl.col('addr_nums_s')=='').then(pl.lit('s_na'))
        .when(pl.col('a')==pl.col('b')).then(pl.lit('eq')).when(pl.col('d').is_in(GEN)).then(pl.lit('gen')).otherwise(pl.lit('other')))
    # number relation subtype for 'other'
    t=t.with_columns(la=pl.col('a').str.len_chars(), lb=pl.col('b').str.len_chars())
    t=t.with_columns(sub=pl.when(pl.col('h')!='other').then(pl.col('h'))
        .when(pl.col('d').is_in([-1,-2])).then(pl.lit('neg12'))
        .when((pl.col('d').abs()%10==0)&(pl.col('la')==pl.col('lb'))).then(pl.lit('tens_sameLen'))
        .when(pl.col('la')<pl.col('lb')).then(pl.lit('shorter'))
        .when(pl.col('la')>pl.col('lb')).then(pl.lit('longer'))
        .otherwise(pl.lit('other_sameLen')))
    # street similarity: alphabetic tokens jaccard
    al=lambda x: pl.col(x).str.replace_all(r'[0-9]+',' ').str.split(' ').list.eval(pl.element().filter(pl.element().str.len_chars()>1)).list.unique()
    t=t.with_columns(sa=al('addr_norm'), sb=al('addr_norm_s'))
    t=t.with_columns(jac=pl.col('sa').list.set_intersection(pl.col('sb')).list.len()/pl.col('sa').list.set_union(pl.col('sb')).list.len().clip(1))
    return t

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
