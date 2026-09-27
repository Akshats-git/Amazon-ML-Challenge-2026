"""Append the runner-up rows (build_ce2.py: ce2_{part}_{c}) to the cross-encoder scoring files (ce_{part}_{c}) as
ce_{part}_{c}_top2.parquet, so a member trained on the argmax files also scores the runner-up candidates."""
import os as _os
import polars as pl
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
D = _os.environ.get('CE_DIR', _os.path.join(SP, 'ce'))
for part in ('P0', 'P1', 'test'):
    for c in ('US', 'India'):
        a = pl.read_parquet(f'{D}/ce_{part}_{c}.parquet')
        b = pl.read_parquet(f'{SP}/ce2_{part}_{c}.parquet')
        m = pl.concat([a, b.select(a.columns)]).unique(['q_row', 's1_row'], keep='first', maintain_order=True)
        m.write_parquet(f'{D}/ce_{part}_{c}_top2.parquet')
        print(part, c, a.height, '+', b.height, '->', m.height)
# France is scored on its argmax rows only
import shutil
shutil.copy(f'{D}/ce_test_France.parquet', f'{D}/ce_test_France_top2.parquet')
