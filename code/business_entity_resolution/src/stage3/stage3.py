"""Stage 3 on argmax rows (US/India). build: per (part, country) numeric feature table. fit: cross-fit P0 <-> P1 pooled
over countries, OOF macro F0.5 with tuned (T1, T2), test p3 = mean of both fold models."""
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


def numcls(d, has_q, has_s):
    return (pl.when(~has_q).then(0).when(~has_s).then(1).when(d == 0).then(2).when(d.is_in(GEN)).then(3)
            .when(d.is_in([-1, -2])).then(4).when(d.abs() <= 21).then(5).otherwise(6))


def logit(c):
    x = pl.col(c).clip(EPS, 1 - EPS)
    return (x / (1 - x)).log()


def build(part, C):
    split = 'test' if part == 'test' else 'train'
    t = pl.read_parquet(f"{SP}/{os.environ.get('GPFX','g')}_{part}_{C}.parquet")
    I = lambda x: pl.col(x).cast(pl.Int32)
    first = lambda x: pl.col(x).str.split(' ').list.first().str.slice(0, 9).cast(pl.Int64, strict=False)
    t = t.with_columns(d_dbl=I('q_dbl') - I('s_dbl'), d_acc=I('q_acc') - I('s_acc'), d_sfx=I('q_numsfx') - I('s_numsfx'),
                       d_frac=I('q_frac') - I('s_frac'), d_rep=I('q_rep') - I('s_rep'),
                       qn=first('addr_nums'), sn=first('addr_nums_s'), is_s3=pl.col('entity_id').str.starts_with('S3').cast(pl.Int8),
                       conf=(pl.col('p') >= 0.5).cast(pl.Int32), q_addr_empty=(pl.col('addr_norm').str.strip_chars() == '').cast(pl.Int8))
    t = t.with_columns(
        n_conf_s=pl.col('conf').sum().over('s1_row') - pl.col('conf'),
        sum_p_s=pl.col('p').sum().over('s1_row') - pl.col('p'),
        n_q_s=pl.len().over('s1_row') - 1,
        rk_s=pl.col('p').rank('ordinal', descending=True).over('s1_row'),
        n_conf_s_same_src=pl.col('conf').sum().over('s1_row', 'is_s3') - pl.col('conf'),
        n_same_nums=pl.col('conf').sum().over('s1_row', 'addr_nums') - pl.col('conf'),
        n_same_name=pl.col('conf').sum().over('s1_row', 'name_norm') - pl.col('conf'),
        n_same_addr=pl.col('conf').sum().over('s1_row', 'addr_norm') - pl.col('conf'),
    )
    cnt = t.filter((pl.col('conf') == 1) & pl.col('qn').is_not_null()).group_by('s1_row', 'qn').agg(c=pl.len())
    tt = t.select('q_row', 's1_row', 'qn', 'conf').join(cnt.rename({'qn': 'cn'}), on='s1_row', how='inner')
    tt = tt.with_columns(c=pl.col('c') - ((pl.col('cn') == pl.col('qn')) & (pl.col('conf') == 1)).cast(pl.UInt32)).filter(pl.col('c') > 0)
    tt = tt.sort(['q_row', 'c', 'cn'], descending=[False, True, False]).group_by('q_row', maintain_order=True).agg(cn=pl.col('cn').first(), cn_c=pl.col('c').first())
    t = t.join(tt, on='q_row', how='left')
    del tt, cnt
    t = t.with_columns(cls_s=numcls(pl.col('qn') - pl.col('sn'), pl.col('qn').is_not_null(), pl.col('sn').is_not_null()),
                       cls_c=numcls(pl.col('qn') - pl.col('cn'), pl.col('qn').is_not_null(), pl.col('cn').is_not_null()),
                       s_eq_c=(pl.col('sn') == pl.col('cn')).cast(pl.Int8), cn_c=pl.col('cn_c').fill_null(0))
    qt = pl.col('name_norm').str.split(' ').list.unique(); st = pl.col('name_norm_s').str.split(' ').list.unique()
    t = t.with_columns(E=qt.list.set_difference(st), M=st.list.set_difference(qt))
    t = t.with_columns(nE=pl.col('E').list.len(), nM=pl.col('M').list.len(), n_shared=qt.list.set_intersection(st).list.len())
    tok = t.filter(pl.col('conf') == 1).select('s1_row', tok=qt).explode('tok').group_by('s1_row', 'tok').agg(tc=pl.len())

    def support(col, name, self_sub):
        e = t.select('q_row', 's1_row', 'conf', col).explode(col).rename({col: 'tok'}).drop_nulls('tok')
        e = e.join(tok, on=['s1_row', 'tok'], how='left').with_columns(tc=pl.col('tc').fill_null(0).cast(pl.Int32))
        if self_sub:
            e = e.with_columns(tc=pl.col('tc') - pl.col('conf'))
        return e.group_by('q_row').agg(**{f'{name}_min': pl.col('tc').min(), f'{name}_max': pl.col('tc').max(), f'{name}_any': (pl.col('tc') > 0).mean()})
    t = t.join(support('E', 'supE', True), on='q_row', how='left').join(support('M', 'supM', False), on='q_row', how='left')
    del tok
    w = pl.col('name_norm_s').str.split(' ').list.eval(pl.element().filter(~pl.element().is_in(LEGAL)))
    a1 = w.list.eval(pl.element().str.slice(0, 1)).list.join('')
    a2 = w.list.eval(pl.element().filter(~pl.element().is_in(STOP)).str.slice(0, 1)).list.join('')
    qn_ = pl.col('name_norm').str.split(' ').list.eval(pl.element().filter(~pl.element().is_in(LEGAL))).list.join('')
    t = t.with_columns(is_acr=((qn_.str.len_chars().is_between(2, 6)) & ((qn_ == a1) | (qn_ == a2)) & (pl.col('name_norm_s').str.split(' ').list.len() >= 2)).cast(pl.Int8))
    t = t.with_columns(q_same_name_all=pl.len().over('name_norm'))
    s1 = pl.read_parquet(f'work/norm/{split}_s1.parquet', columns=['name_norm', 'country']).with_row_index('s1_row').with_columns(pl.col('s1_row').cast(pl.Int32))
    if split == 'train':
        pool = np.load(f'work/pools/{part}_{C}.npz')['s1_rows']
        s1 = s1.filter(pl.col('s1_row').is_in(pl.Series(pool).implode()))
    else:
        s1 = s1.filter(pl.col('country') == C)
    nf = s1.group_by('name_norm').agg(s1_same_name=pl.len())
    t = t.join(nf, on='name_norm', how='left').with_columns(pl.col('s1_same_name').fill_null(0))
    t = t.with_columns(lp=logit('p'), lp2=logit('p2nd'), gap=pl.col('p') - pl.col('p2nd').clip(0, 1),
                       lx1=logit('p_x1'), lx2=logit('p_x2'), lx3=logit('p_x3'), lx4=logit('p_x4'),
                       is_india=pl.lit(1 if C == 'India' else 0, dtype=pl.Int8))
    if 'p_s2' in t.columns:
        t = t.with_columns(lps2=logit('p_s2'))
    keep = ['q_row', 's1_row', 'p', 'p2nd', 'p_s2', 'lps2'] + (['ok', 'has_true'] if split == 'train' else []) + FEATS
    t.select([k for k in keep if k in t.columns]).write_parquet(f"{SP}/{os.environ.get('FPFX','f')}_{part}_{C}.parquet")
    print('built', part, C, t.height)


FEATS = ['lp', 'lp2', 'gap', 'lx1', 'lx2', 'lx3', 'lx4', 'n_cand', 'is_india',
         'n_conf_s', 'sum_p_s', 'n_q_s', 'rk_s', 'n_conf_s_same_src', 'n_same_nums', 'n_same_name', 'n_same_addr',
         'cls_s', 'cls_c', 's_eq_c', 'cn_c', 'nE', 'nM', 'n_shared', 'supE_min', 'supE_max', 'supE_any', 'supM_min', 'supM_max', 'supM_any',
         'is_acr', 'q_same_name_all', 's1_same_name', 'q_addr_empty', 'is_s3',
         'd_dbl', 'q_tri', 'd_acc', 'd_sfx', 'd_frac', 'd_rep', 'q_brack', 'q_paren', 'q_hash']
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6, seed=7)


def _load(part, cols, cs=('US', 'India')):
    return pl.concat([pl.read_parquet(f'{SP}/{os.environ.get("FPFX", "f")}_{part}_{c}.parquet', columns=cols) for c in cs])


def fit(feats):
    import gc
    from ber import io
    from ber.decide import assign, tune_thresholds
    FP = os.environ.get('FPFX', 'f'); TC = os.environ.get('TEST_C', 'US,India').split(','); OUT = os.environ.get('OUT', 'p3')
    t0 = time.time()
    meta = ['q_row', 's1_row', 'p', 'p2nd', 'ok', 'has_true', 'is_india']
    if 'p_s2' in pl.read_parquet_schema(f'{SP}/{FP}_P0_US.parquet'):
        meta = meta + ['p_s2']
    pt = {c: np.zeros(pl.scan_parquet(f'{SP}/{FP}_test_{c}.parquet').select(pl.len()).collect().item()) for c in TC}
    oof = {}
    for tr, ev in (('P0', 'P1'), ('P1', 'P0')):
        a = _load(tr, feats + ['ok'])
        y = a['ok'].cast(pl.Int8).to_numpy()
        X = a.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        del a; gc.collect()
        hold = np.random.default_rng(0).random(len(y)) < 0.1
        dtr = lgb.Dataset(X[~hold], y[~hold], feature_name=feats, free_raw_data=True)
        dho = lgb.Dataset(X[hold], y[hold], reference=dtr)
        m = lgb.train(PARAMS, dtr, 4000, valid_sets=[dho], callbacks=[lgb.early_stopping(100, verbose=False)])
        del X, y, dtr, dho; gc.collect()
        b = _load(ev, sorted(set(meta + feats)))
        p3 = m.predict(b.select(pl.col(feats).cast(pl.Float32)).to_numpy(), num_iteration=m.best_iteration)
        yb = b['ok'].cast(pl.Int8).to_numpy()
        ll = lambda q: -np.mean(yb * np.log(np.clip(q, 1e-7, 1)) + (1 - yb) * np.log(np.clip(1 - q, 1e-7, 1)))
        print(f'{tr}->{ev}: best_iter {m.best_iteration}  logloss p {ll(b["p"].to_numpy()):.5f} -> p3 {ll(p3):.5f}  ({time.time() - t0:.0f}s)', flush=True)
        oof[ev] = b.select(meta).with_columns(p3=pl.Series(p3.astype(np.float32)))
        del b; gc.collect()
        for c in TC:
            Xt = pl.read_parquet(f'{SP}/{FP}_test_{c}.parquet', columns=feats).select(pl.col(feats).cast(pl.Float32)).to_numpy()
            pt[c] += m.predict(Xt, num_iteration=m.best_iteration) / 2
            del Xt
        imp = sorted(zip(m.feature_importance('gain'), feats), reverse=True); tot = sum(g for g, _ in imp)
        print('  top gain:', ', '.join(f'{f} {g / tot:.3f}' for g, f in imp[:16]), flush=True)
        m.save_model(f'{SP}/{OUT}_m3_{tr}.txt')
    allt = pl.concat([oof['P0'], oof['P1']])
    gt = io.gt_rows()
    res = {}
    for C in ('US', 'India', None):
        cs = ('US', 'India') if C is None else (C,)
        uni = np.concatenate([np.load(f'work/pools/{p}_{c}.npz')['eval_s1'] for p in ('P0', 'P1') for c in cs])
        qa = np.concatenate([np.load(f'work/pools/{p}_{c}.npz')['q_rows'] for p in ('P0', 'P1') for c in cs])
        truth = gt.filter(pl.col('s1_row').is_in(pl.Series(uni).implode()) & pl.col('q_row').is_in(pl.Series(qa).implode()))
        sub = allt if C is None else allt.filter(pl.col('is_india') == (1 if C == 'India' else 0))
        base = sub.with_columns(pt=pl.col('p_s2') if 'p_s2' in sub.columns else pl.col('p'))
        base = base.filter((pl.col('pt') - pl.col('p2nd')) >= 1e-6)  # ties of the stage-2 blend, as compose.py does
        for col in ('p', 'p3'):
            top = (base.with_columns(p=pl.col(col)).select('q_row', 's1_row', 'p', 'ok', 'has_true')
                   .with_columns(rk=pl.col('p').rank('ordinal', descending=True).over('s1_row')))
            (f, th), grid = tune_thresholds(top, truth, uni)
            res[(C, col)] = (f, th, {(a, b): v for a, b, v in grid})
            print(f'{C or "pooled"} F0.5 {col}: {f:.5f} at {th}', flush=True)
    th = res[(None, 'p3')][1]
    for C in ('US', 'India'):
        print(f'  {C} at pooled p3 thresholds {th}: {res[(C, "p3")][2][th]:.5f}  (p at its pooled {res[(None, "p")][1]}: {res[(C, "p")][2][res[(None, "p")][1]]:.5f})')
    tcols = ['q_row', 's1_row', 'p', 'p2nd'] + (['p_s2'] if 'p_s2' in meta else [])
    tk = pl.concat([pl.read_parquet(f'{SP}/{FP}_test_{c}.parquet', columns=tcols).with_columns(p3=pl.Series(pt[c].astype(np.float32))) for c in TC])
    tk.write_parquet(f'{SP}/{OUT}_test.parquet')
    json.dump(dict(T1=th[0], T2=th[1], F=res[(None, 'p3')][0], F_p=res[(None, 'p')][0]), open(f'{SP}/{OUT}_thresholds.json', 'w'))
    allt.write_parquet(f'{SP}/{OUT}_oof.parquet')
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(sys.argv[2], sys.argv[3])
    else:
        fit(sys.argv[2].split(',') if len(sys.argv) > 2 else FEATS)
