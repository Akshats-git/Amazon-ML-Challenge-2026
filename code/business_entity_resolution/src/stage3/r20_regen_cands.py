"""candidate_pairs.tsv = original blocking candidates (work/cands/test_*) U rescue-scored pairs (rf_test_*); streamed check vs a matching file.
usage: regen_cands.py OUT_TSV MATCH_TSV COUNTRIES(e.g. India,US)"""
import os as _os
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
R = _os.environ.get('R20_DIR', _os.path.join(SP, 'r20'))
_os.makedirs(R, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os, sys, numpy as np, polars as pl
os.chdir(ROOT); pass
from ber import io
from ber.submit import write_candidates, check_candidates
out, match, cs = sys.argv[1], sys.argv[2], [c for c in sys.argv[3].split(',') if c]
orig=pl.concat([pl.read_parquet(f'work/cands/test_{c}.parquet',columns=['q_row','s1_row']) for c in ('US','India','France')])
print('original pairs', f'{orig.height:,}')
add=[pl.read_parquet(f'{R}/rf_test_{c}.parquet',columns=['q_row','s1_row']) for c in cs]
allp=pl.concat([orig]+add).unique() if add else orig
print('rescue-scored pairs', f'{sum(a.height for a in add):,}', ' union', f'{allp.height:,}', ' new', f'{allp.height-orig.height:,}')
s1_ids=pl.read_parquet(io.norm_path('test',1),columns=['entity_id'])['entity_id']
q_ids=io.load_norm('test','q',columns=['entity_id'])['entity_id']
k=write_candidates(out, s1_ids, q_ids, allp['s1_row'].to_numpy(), allp['q_row'].to_numpy())
print('written pairs', f'{k:,}')
check_candidates(out, match, s1_ids.to_list())
