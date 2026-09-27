"""(R20) Rescue cross-encoder: trained on one pool's rescue candidates (r20_build_rce.py), scores the other pool + test.
Cross-encoder on raw record text (query 'name | address' vs S1 'name | address'), trained on one pool's argmax rows and
scored on the other pool + test. usage: ce_train.py TRAIN_POOL [N_TRAIN]   (env: CE_MODEL, CE_BS, CE_LEN, CE_LR, CE_DIR)"""
import os as _os, sys as _sys
ROOT = _os.environ.get('BER_ROOT', _os.getcwd())
SP = _os.environ.get('S3_DIR', _os.path.join(ROOT, 'work', 'stage3'))
_os.makedirs(SP, exist_ok=True)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, sys, time, math
import numpy as np, polars as pl, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
D = os.environ.get('CE_DIR', _os.path.join(SP, 'r20', 'rce'))
MODEL = os.environ.get('CE_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
BS = int(os.environ.get('CE_BS', '64')); ML = int(os.environ.get('CE_LEN', '96')); LR = float(os.environ.get('CE_LR', '4e-5'))
EBS = int(os.environ.get('CE_EBS', '512'))
tr = sys.argv[1]; ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ev = 'P1' if tr == 'P0' else 'P0'
tag = os.environ.get('CE_TAG', 'ce')
SCORE_SFX = os.environ.get('CE_SCORE_SFX', '')  # '_top2': score the files with the runner-up rows appended (ce_rows_top2.py)
dev = 'cuda'
DT = torch.bfloat16 if os.environ.get('CE_BF16', '0') == '1' else torch.float16
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, dtype=torch.float32).to(dev)
if os.environ.get('CE_FREEZE_EMB', '1') == '1':
    for n_, p_ in model.named_parameters():
        if 'word_embeddings' in n_:
            p_.requires_grad = False
t0 = time.time()
trn = pl.concat([pl.read_parquet(f'{D}/rce_{tr}_{c}.parquet', columns=['a', 'b', 'ok']) for c in ('US', 'India')])
trn = trn.sample(fraction=1.0, shuffle=True, seed=0)
if ntr:
    trn = trn.head(ntr)
nho = min(20000, trn.height // 20)
ho, trn = trn.head(nho), trn.slice(nho)
REP = int(os.environ.get('CE_REP', '1'))
trn = pl.concat([trn.sample(fraction=1.0, shuffle=True, seed=k) for k in range(REP)])
print(f'train {trn.height:,}  holdout {ho.height:,}  model {MODEL}', flush=True)


def collate(batch):
    a, b, y = zip(*batch)
    enc = tok(list(a), list(b), truncation=True, max_length=ML, padding=True, return_tensors='pt')
    enc['labels'] = torch.tensor(y, dtype=torch.float32)
    return enc


def rows(df, lab=True):
    y = df['ok'].cast(pl.Float32).to_list() if lab else [0.0] * df.height
    return list(zip(df['a'].to_list(), df['b'].to_list(), y))


dl = DataLoader(rows(trn), batch_size=BS, shuffle=False, collate_fn=collate, num_workers=int(os.environ.get('CE_NW', '2')))
opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=LR, weight_decay=0.01)
steps = len(dl)
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
lossf = torch.nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler()
model.train()
run = 0.0
for i, enc in enumerate(dl):
    enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
    y = enc.pop('labels')
    with torch.autocast('cuda', dtype=DT):
        out = model(**enc).logits.squeeze(-1)
    loss = lossf(out.float(), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sch.step()
    run = 0.98 * run + 0.02 * loss.item() if i else loss.item()
    if i % 100 == 0:
        print(f'step {i}/{steps} loss {run:.4f}  {(i + 1) * BS / (time.time() - t0):.0f} samples/s', flush=True)
model.eval()


@torch.no_grad()
def score(df):
    order = np.argsort((df['a'].str.len_chars() + df['b'].str.len_chars()).to_numpy())  # length bucketing
    a, b = df['a'].to_numpy()[order], df['b'].to_numpy()[order]
    out = np.empty(df.height, np.float32)
    for s in range(0, df.height, EBS):
        enc = tok(list(a[s:s + EBS]), list(b[s:s + EBS]), truncation=True, max_length=ML, padding=True, return_tensors='pt').to(dev)
        with torch.autocast('cuda', dtype=DT):
            out[order[s:s + EBS]] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    return out


h = score(ho)
yh = ho['ok'].cast(pl.Float32).to_numpy()
ph = 1 / (1 + np.exp(-h))
print(f'holdout logloss {-np.mean(yh * np.log(np.clip(ph, 1e-7, 1)) + (1 - yh) * np.log(np.clip(1 - ph, 1e-7, 1))):.4f}  acc {((ph > 0.5) == yh).mean():.4f}  ({time.time() - t0:.0f}s)', flush=True)
os.makedirs(f'{D}/out', exist_ok=True)
for part, c in [(ev, 'US'), (ev, 'India'), ('test', 'US'), ('test', 'India')]:
    df = pl.read_parquet(f'{D}/rce_{part}_{c}.parquet', columns=['q_row', 's1_row', 'a', 'b'])
    sc = score(df)
    df.select('q_row', 's1_row').with_columns(ce=pl.Series(sc)).write_parquet(f'{D}/out/{tag}_{tr}_{part}_{c}.parquet')
    print(f'scored {part} {c} {df.height:,}  ({time.time() - t0:.0f}s)', flush=True)
print('CE_DONE')
