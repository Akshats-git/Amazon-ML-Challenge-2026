"""Top-2 re-rank stage: one row per (query, candidate) for the two best candidates of the stage-2 blend. Consensus over the
OTHER queries' argmax rows (as stage 3), plus the competing row's S1 statistics. Cross-fit P0 <-> P1; per query the row with
the highest p3 is the link candidate; (T1, T2) tuned on pooled OOF. build: python stack3.py build PART C; fit: python stack3.py fit"""
import os as _os, sys as _sys
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
_os.makedirs(SP, exist_ok=True)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, time, json
import numpy as np, polars as pl, lightgbm as lgb
os.chdir(ROOT); 
GEN = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
LEGAL = ['inc','lc','llc','corp','co','ltd','lp','llp','pc','plc','pllc','sarl','sas','sasu','sa','eurl','sci','snc','pvt','private','limited','l','c']
STOP = ['de','du','des','la','le','les','l','d','et','and','of','the','a']
EPS = 1e-6
HP = os.environ.get('HPFX', 'h'); KP = os.environ.get('KPFX', 'k'); OUT = os.environ.get('OUT', 'p5')
NT = int(os.environ.get('NT', '16'))


def numcls(d, has_q, has_s):
    return (pl.when(~has_q).then(0).when(~has_s).then(1).when(d == 0).then(2).when(d.is_in(GEN)).then(3)
            .when(d.is_in([-1, -2])).then(4).when(d.abs() <= 21).then(5).otherwise(6))


def logit(c):
    x = pl.col(c).clip(EPS, 1 - EPS)
    return (x / (1 - x)).log()


def build(part, C):
    split = 'test' if part == 'test' else 'train'
    t = pl.read_parquet(f'{SP}/{HP}_{part}_{C}.parquet')
    I = lambda x: pl.col(x).cast(pl.Int32)
    first = lambda x: pl.col(x).str.split(' ').list.first().str.slice(0, 9).cast(pl.Int64, strict=False)
    t = t.with_columns(d_dbl=I('q_dbl') - I('s_dbl'), d_acc=I('q_acc') - I('s_acc'), d_sfx=I('q_numsfx') - I('s_numsfx'),
                       d_frac=I('q_frac') - I('s_frac'), d_rep=I('q_rep') - I('s_rep'),
                       qn=first('addr_nums'), sn=first('addr_nums_s'), is_s3=pl.col('entity_id').str.starts_with('S3').cast(pl.Int8),
                       conf=(pl.col('p') >= 0.5).cast(pl.Int32), q_addr_empty=(pl.col('addr_norm').str.strip_chars() == '').cast(pl.Int8),
                       am=(pl.col('r') == 1).cast(pl.Int32))
    t = t.with_columns(selfc=pl.col('conf') * pl.col('am'), selfp=pl.col('p') * pl.col('am'))
    a = t.filter(pl.col('r') == 1)
    st = a.group_by('s1_row').agg(n_conf_s=pl.col('conf').sum(), sum_p_s=pl.col('p').sum(), n_q_s=pl.len())
    t = t.join(st, on='s1_row', how='left').with_columns([pl.col(x).fill_null(0) for x in ('n_conf_s', 'sum_p_s', 'n_q_s')])
    t = t.with_columns(n_conf_s=pl.col('n_conf_s') - pl.col('selfc'), sum_p_s=pl.col('sum_p_s') - pl.col('selfp'), n_q_s=pl.col('n_q_s') - pl.col('am'))
    for key, name in (('is_s3', 'n_conf_s_same_src'), ('addr_nums', 'n_same_nums'), ('name_norm', 'n_same_name'), ('addr_norm', 'n_same_addr')):
        c = a.group_by('s1_row', key).agg(pl.col('conf').sum().alias(name))
        t = t.join(c, on=['s1_row', key], how='left').with_columns((pl.col(name).fill_null(0) - pl.col('selfc')).alias(name))
    # rank of this row's p among the S1's argmax rows (self excluded)
    ka = np.sort((a['s1_row'].to_numpy().astype(np.int64) << 32) | (0xFFFFFFFF - a['p'].to_numpy().astype(np.float32).view(np.uint32).astype(np.int64)))
    s = t['s1_row'].to_numpy().astype(np.int64)
    key = (s << 32) | (0xFFFFFFFF - t['p'].to_numpy().astype(np.float32).view(np.uint32).astype(np.int64))
    t = t.with_columns(rk_s=pl.Series((np.searchsorted(ka, key, 'left') - np.searchsorted(ka, s << 32, 'left') + 1).astype(np.int32)))
    # consensus first number over other confident argmax rows
    cnt = a.filter((pl.col('conf') == 1) & pl.col('qn').is_not_null()).group_by('s1_row', 'qn').agg(c=pl.len())
    tt = t.select('q_row', 's1_row', 'qn', 'selfc').join(cnt.rename({'qn': 'cn'}), on='s1_row', how='inner')
    tt = tt.with_columns(c=pl.col('c').cast(pl.Int32) - ((pl.col('cn') == pl.col('qn')) & (pl.col('selfc') == 1)).cast(pl.Int32)).filter(pl.col('c') > 0)
    tt = tt.sort(['q_row', 's1_row', 'c', 'cn'], descending=[False, False, True, False]).group_by('q_row', 's1_row', maintain_order=True).agg(cn=pl.col('cn').first(), cn_c=pl.col('c').first())
    t = t.join(tt, on=['q_row', 's1_row'], how='left')
    del tt, cnt
    t = t.with_columns(cls_s=numcls(pl.col('qn') - pl.col('sn'), pl.col('qn').is_not_null(), pl.col('sn').is_not_null()),
                       cls_c=numcls(pl.col('qn') - pl.col('cn'), pl.col('qn').is_not_null(), pl.col('cn').is_not_null()),
                       s_eq_c=(pl.col('sn') == pl.col('cn')).cast(pl.Int8), cn_c=pl.col('cn_c').fill_null(0))
    qt = pl.col('name_norm').str.split(' ').list.unique(); stt = pl.col('name_norm_s').str.split(' ').list.unique()
    t = t.with_columns(E=qt.list.set_difference(stt), M=stt.list.set_difference(qt))
    t = t.with_columns(nE=pl.col('E').list.len(), nM=pl.col('M').list.len(), n_shared=qt.list.set_intersection(stt).list.len())
    tok = a.filter(pl.col('conf') == 1).select('s1_row', tok=qt).explode('tok').group_by('s1_row', 'tok').agg(tc=pl.len())

    def support(col, name, self_sub):
        e = t.select('q_row', 's1_row', 'selfc', col).explode(col).rename({col: 'tok'}).drop_nulls('tok')
        e = e.join(tok, on=['s1_row', 'tok'], how='left').with_columns(tc=pl.col('tc').fill_null(0).cast(pl.Int32))
        if self_sub:
            e = e.with_columns(tc=pl.col('tc') - pl.col('selfc'))
        return e.group_by('q_row', 's1_row').agg(**{f'{name}_min': pl.col('tc').min(), f'{name}_max': pl.col('tc').max(), f'{name}_any': (pl.col('tc') > 0).mean()})
    t = t.join(support('E', 'supE', True), on=['q_row', 's1_row'], how='left').join(support('M', 'supM', False), on=['q_row', 's1_row'], how='left')
    del tok, a
    w = pl.col('name_norm_s').str.split(' ').list.eval(pl.element().filter(~pl.element().is_in(LEGAL)))
    a1 = w.list.eval(pl.element().str.slice(0, 1)).list.join('')
    a2 = w.list.eval(pl.element().filter(~pl.element().is_in(STOP)).str.slice(0, 1)).list.join('')
    qn_ = pl.col('name_norm').str.split(' ').list.eval(pl.element().filter(~pl.element().is_in(LEGAL))).list.join('')
    t = t.with_columns(is_acr=((qn_.str.len_chars().is_between(2, 6)) & ((qn_ == a1) | (qn_ == a2)) & (pl.col('name_norm_s').str.split(' ').list.len() >= 2)).cast(pl.Int8))
    qnm = t.filter(pl.col('r') == 1).group_by('name_norm').agg(q_same_name_all=pl.len())
    t = t.join(qnm, on='name_norm', how='left').with_columns(pl.col('q_same_name_all').fill_null(1))
    s1 = pl.read_parquet(f'work/norm/{split}_s1.parquet', columns=['name_norm', 'country']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
    if split == 'train':
        pool = np.load(f'work/pools/{part}_{C}.npz')['s1_rows']
        s1 = s1.filter(pl.col('s1_row').is_in(pl.Series(pool).implode()))
    else:
        s1 = s1.filter(pl.col('country') == C)
    nf = s1.group_by('name_norm').agg(s1_same_name=pl.len())
    t = t.join(nf.rename({'name_norm': 'name_norm_s'}), on='name_norm_s', how='left').with_columns(pl.col('s1_same_name').fill_null(0))
    t = t.with_columns(lp=logit('p'), lpo=logit('p_other'), dp=pl.col('p') - pl.col('p_other').fill_null(0),
                       lx1=logit('p_x1'), lx2=logit('p_x2'), lx3=logit('p_x3'), lx4=logit('p_x4'),
                       is_india=pl.lit(1 if C == 'India' else 0, dtype=pl.Int8))
    # the competing row's S1 statistics (record-count prior between twins)
    for x in ('n_conf_s', 'sum_p_s', 'n_q_s', 'n_same_name', 'cls_s', 'nE', 'nM', 'n_shared', 's1_same_name', 'supM_any'):
        t = t.with_columns(pl.when(pl.len().over('q_row') == 2).then(pl.col(x).cast(pl.Float64).sum().over('q_row') - pl.col(x).cast(pl.Float64)).otherwise(None).alias(f'o_{x}'))
    keep = ['q_row', 's1_row', 'p'] + (['y', 'has_true'] if split == 'train' else []) + FEATS
    t.select(keep).write_parquet(f'{SP}/{KP}_{part}_{C}.parquet')
    print('built', part, C, t.height, flush=True)


FEATS = ['r', 'lp', 'lpo', 'dp', 'lx1', 'lx2', 'lx3', 'lx4', 'n_cand', 'is_india',
         'n_conf_s', 'sum_p_s', 'n_q_s', 'rk_s', 'n_conf_s_same_src', 'n_same_nums', 'n_same_name', 'n_same_addr',
         'cls_s', 'cls_c', 's_eq_c', 'cn_c', 'nE', 'nM', 'n_shared', 'supE_min', 'supE_max', 'supE_any', 'supM_min', 'supM_max', 'supM_any',
         'is_acr', 'q_same_name_all', 's1_same_name', 'q_addr_empty', 'is_s3',
         'd_dbl', 'q_tri', 'd_acc', 'd_sfx', 'd_frac', 'd_rep', 'q_brack', 'q_paren', 'q_hash',
         'o_n_conf_s', 'o_sum_p_s', 'o_n_q_s', 'o_n_same_name', 'o_cls_s', 'o_nE', 'o_nM', 'o_n_shared', 'o_s1_same_name', 'o_supM_any']
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=127, min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=NT, seed=7)


def choose(df, col):
    """per query the row with the highest col; ties (|diff| < EPS) dropped; rank within S1 by col."""
    df = df.sort(['q_row', col], descending=[False, True])
    best = df.group_by('q_row', maintain_order=True).agg(pl.all().first(), second=pl.col(col).slice(1, 1).first())
    best = best.filter(pl.col('second').is_null() | ((pl.col(col) - pl.col('second')) >= EPS))
    return best.with_columns(p=pl.col(col)).with_columns(rk=pl.col('p').rank('ordinal', descending=True).over('s1_row'))


def fit(feats):
    import gc
    from ber import io
    from ber.decide import tune_thresholds
    t0 = time.time()
    load = lambda part, cols, cs=('US', 'India'): pl.concat([pl.read_parquet(f'{SP}/{KP}_{part}_{c}.parquet', columns=cols).with_columns(C=pl.lit(c)) for c in cs])
    meta = ['q_row', 's1_row', 'p', 'y', 'has_true', 'is_india']
    pt = {c: np.zeros(pl.scan_parquet(f'{SP}/{KP}_test_{c}.parquet').select(pl.len()).collect().item()) for c in ('US', 'India')}
    oof = {}
    for tr, ev in (('P0', 'P1'), ('P1', 'P0')):
        a = load(tr, feats + ['y', 'q_row'])
        y = a['y'].cast(pl.Int8).to_numpy()
        hold = (a['q_row'].hash(7) % 10 == 0).to_numpy()  # query-level holdout
        X = a.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        del a; gc.collect()
        dtr = lgb.Dataset(X[~hold], y[~hold], feature_name=feats, free_raw_data=True)
        dho = lgb.Dataset(X[hold], y[hold], reference=dtr)
        m = lgb.train(PARAMS, dtr, 5000, valid_sets=[dho], callbacks=[lgb.early_stopping(100, verbose=False)])
        del X, y, dtr, dho; gc.collect()
        b = load(ev, sorted(set(meta + feats)))
        p5 = m.predict(b.select(pl.col(feats).cast(pl.Float32)).to_numpy(), num_iteration=m.best_iteration)
        print(f'{tr}->{ev}: best_iter {m.best_iteration} ({time.time() - t0:.0f}s)', flush=True)
        oof[ev] = b.select(meta).with_columns(p5=pl.Series(p5.astype(np.float32)))
        del b; gc.collect()
        for c in ('US', 'India'):
            Xt = pl.read_parquet(f'{SP}/{KP}_test_{c}.parquet', columns=feats).select(pl.col(feats).cast(pl.Float32)).to_numpy()
            pt[c] += m.predict(Xt, num_iteration=m.best_iteration) / 2
            del Xt
        imp = sorted(zip(m.feature_importance('gain'), feats), reverse=True); tot = sum(g for g, _ in imp)
        print('  top gain:', ', '.join(f'{f} {g / tot:.3f}' for g, f in imp[:18]), flush=True)
        m.save_model(f'{SP}/{OUT}_m_{tr}.txt')
    allt = pl.concat([oof['P0'].with_columns(pool=pl.lit('P0')), oof['P1'].with_columns(pool=pl.lit('P1'))])
    allt.write_parquet(f'{SP}/{OUT}_oof_rows.parquet')
    gt = io.gt_rows()
    res = {}
    for C in ('US', 'India', None):
        cs = ('US', 'India') if C is None else (C,)
        uni = np.concatenate([np.load(f'work/pools/{p}_{c}.npz')['eval_s1'] for p in ('P0', 'P1') for c in cs])
        qa = np.concatenate([np.load(f'work/pools/{p}_{c}.npz')['q_rows'] for p in ('P0', 'P1') for c in cs])
        truth = gt.filter(pl.col('s1_row').is_in(pl.Series(uni).implode()) & pl.col('q_row').is_in(pl.Series(qa).implode()))
        sub = allt if C is None else allt.filter(pl.col('is_india') == (1 if C == 'India' else 0))
        for col in ('p', 'p5'):
            # distractor queries appear in both pools: choose per (pool, query)
            parts = []
            for pool in ('P0', 'P1'):
                ch = choose(sub.filter(pl.col('pool') == pool), col)
                parts.append(ch)
            top = pl.concat(parts).with_columns(ok=pl.col('y')).select('q_row', 's1_row', 'p', 'ok', 'has_true', 'rk')
            top = top.with_columns(rk=pl.col('p').rank('ordinal', descending=True).over('s1_row'))
            (f, th), grid = tune_thresholds(top, truth, uni)
            res[(C, col)] = (f, th, {(x, z): v for x, z, v in grid})
            print(f'{C or "pooled"} F0.5 {col}: {f:.5f} at {th}', flush=True)
    th = res[(None, 'p5')][1]
    for C in ('US', 'India'):
        print(f'  {C} at pooled p5 thresholds {th}: {res[(C, "p5")][2][th]:.5f}')
    tk = pl.concat([pl.read_parquet(f'{SP}/{KP}_test_{c}.parquet', columns=['q_row', 's1_row', 'r', 'p']).with_columns(p5=pl.Series(pt[c].astype(np.float32))) for c in ('US', 'India')])
    tk.write_parquet(f'{SP}/{OUT}_test_rows.parquet')
    json.dump(dict(T1=th[0], T2=th[1], F=res[(None, 'p5')][0], F_p=res[(None, 'p')][0]), open(f'{SP}/{OUT}_thresholds.json', 'w'))
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(sys.argv[2], sys.argv[3])
    else:
        fit(sys.argv[2].split(',') if len(sys.argv) > 2 else FEATS)
