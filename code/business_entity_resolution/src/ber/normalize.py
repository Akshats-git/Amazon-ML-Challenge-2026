"""Section 7.1 normalization, identical for every source/split/country. Parallel over row chunks.

Output work/norm/{split}_s{src}.parquet (row order = file order) with columns:
  entity_id, country, name_norm, name_core, name_sq, legal, alias_l, alias_r,
  nonlatin, web, alias, idtag, addr_norm, addr_null, addr_nums
"""
import json
import re
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

from . import config as C
from . import io

# outside U+0000-U+024F, ignoring typographic punctuation/currency (U+2000-U+20CF), which is not a script
RE_NONLATIN = re.compile("[^\u0000-ɏ -⃏]")
RE_WEB = re.compile(r"(?i)(\.(com|c0m|net|org|in|co|fr)\b|www\.|^@)")
RE_WEB_SUFFIX = re.compile(r"(?i)\.(com|c0m|net|org|in|co|fr)\b")
RE_WWW = re.compile(r"(?i)www\.")
RE_ALIAS = re.compile(r"(?i)\b(dba|d/b/a|t/a|a/k/a|aka|doing business as|trading as)\b")
RE_IDTAG = re.compile(r"(?i)\(id:\s*\d+\)|#\d{3,}")
RE_NONALNUM = re.compile(r"[^a-z0-9]+")
RE_OCR = re.compile(r"(?<=[a-z])[015]|[015](?=[a-z])")
OCR_MAP = {"0": "o", "1": "l", "5": "s"}
RE_DOUBLE = re.compile(r"([a-z])\1+")
RE_DIGITS = re.compile(r"\d+")
ADDR_NULL = {"", "n/a", "na", "null", "none"}

# generic legal-form tokens (compared after the doubled-letter collapse) -> canonical form
LEGAL = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "lc": "lc",  # llc
    "ltd": "ltd", "limited": "ltd",
    "lp": "lp",  # lp, llp
    "plc": "plc",  # plc, pllc
    "pc": "pc",
    "pvt": "pvt", "private": "pvt",
    "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "scp": "scp", "selarl": "selarl", "ei": "ei", "gmbh": "gmbh", "ag": "ag",
}

# Hand maps: only for the single Step-7 A/B check (USE_HAND_MAPS=1). Never used in the final pipeline.
HAND_MAPS = {
    "street": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd", "drive": "dr", "lane": "ln",
    "court": "ct", "place": "pl", "highway": "hwy", "parkway": "pkwy", "suite": "ste", "north": "n", "south": "s",
    "east": "e", "west": "w", "nagar": "ngr", "circle": "cir", "terrace": "ter", "square": "sq",
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "conecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawai": "hi", "idaho": "id",
    "ilinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "masachusets": "ma", "michigan": "mi", "minesota": "mn", "misisipi": "ms",
    "misouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pensylvania": "pa", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "wisconsin": "wi", "wyoming": "wy", "tenesee": "tn",
    "maharashtra": "mh", "karnataka": "ka", "tamil": "tn", "telangana": "tg", "gujarat": "gj", "kerala": "kl",
    "rajasthan": "rj", "punjab": "pb", "haryana": "hr", "bihar": "br", "odisha": "od", "delhi": "dl",
}

_NAME_DICT: dict = {}
_ADDR_DICT: dict = {}


def load_dicts():
    global _NAME_DICT, _ADDR_DICT
    np_, ap = C.DICT_DIR / "name.json", C.DICT_DIR / "addr.json"
    _NAME_DICT = json.loads(np_.read_text()) if np_.exists() else {}
    _ADDR_DICT = json.loads(ap.read_text()) if ap.exists() else {}
    return len(_NAME_DICT), len(_ADDR_DICT)


def _ocr_token(t: str) -> str:
    if t.isdigit() or t.isalpha():
        return t
    for _ in range(4):
        t2 = RE_OCR.sub(lambda m: OCR_MAP[m.group(0)], t)
        if t2 == t:
            break
        t = t2
    return t


def _basic(s: str) -> str:
    """Steps 4, 5: unidecode, lowercase, & -> and, non-alnum -> space."""
    if not s.isascii():
        s = unidecode(s)
    s = s.lower().replace("&", " and ")
    return RE_NONALNUM.sub(" ", s)


def name_tokens_pre(raw: str) -> list:
    """Steps 3-7 (no dictionary): cleaned name tokens."""
    s = RE_IDTAG.sub(" ", raw)
    s = RE_WWW.sub("", s).strip()
    if s.startswith("@"):
        s = s[1:]
    s = RE_WEB_SUFFIX.sub("", s)
    s = _basic(s)
    toks = [_ocr_token(t) if any(c.isalpha() for c in t) else t for t in s.split()]
    return [RE_DOUBLE.sub(r"\1", t) for t in toks]


def _name_post(toks: list, nonlatin: bool):
    """Steps 8, 11, 12: dictionary, legal canonicalization, core, squashed."""
    if nonlatin and _NAME_DICT:
        toks = " ".join(_NAME_DICT.get(t, t) for t in toks).split()
    if C.USE_HAND_MAPS:
        toks = [HAND_MAPS.get(t, t) for t in toks]
    toks = [LEGAL.get(t, t) for t in toks]
    legal = sorted({t for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL] or toks
    core_s = " ".join(core)
    return " ".join(toks), core_s, core_s.replace(" ", ""), " ".join(legal)


def addr_tokens_pre(raw: str) -> list:
    s = _basic(raw)
    return [RE_DOUBLE.sub(r"\1", t) for t in s.split()]


def norm_record(name: str, addr: str):
    nonlatin = bool(RE_NONLATIN.search(name) or RE_NONLATIN.search(addr))
    web = bool(RE_WEB.search(name))
    alias = bool(RE_ALIAS.search(name))
    idtag = bool(RE_IDTAG.search(name))
    name_norm, name_core, name_sq, legal = _name_post(name_tokens_pre(name), nonlatin)
    alias_l = alias_r = ""
    if alias:
        m = RE_ALIAS.search(name)
        alias_l = _name_post(name_tokens_pre(name[: m.start()]), nonlatin)[1]
        alias_r = _name_post(name_tokens_pre(name[m.end():]), nonlatin)[1]
    addr_null = addr.strip().lower() in ADDR_NULL
    atoks = [] if addr_null else addr_tokens_pre(addr)
    if nonlatin and _ADDR_DICT:
        atoks = " ".join(_ADDR_DICT.get(t, t) for t in atoks).split()
    if C.USE_HAND_MAPS:
        atoks = [HAND_MAPS.get(t, t) for t in atoks]
    # drop standalone "n a" (from N/A)
    out, i = [], 0
    while i < len(atoks):
        if atoks[i] == "n" and i + 1 < len(atoks) and atoks[i + 1] == "a":
            i += 2
            continue
        out.append(atoks[i])
        i += 1
    addr_norm = " ".join(out)
    nums = [d.lstrip("0") or "0" for d in RE_DIGITS.findall(addr_norm)]
    return (name_norm, name_core, name_sq, legal, alias_l, alias_r, nonlatin, web, alias, idtag,
            addr_norm, addr_null or not addr_norm, " ".join(nums))


COLS = ["name_norm", "name_core", "name_sq", "legal", "alias_l", "alias_r", "nonlatin", "web", "alias", "idtag",
        "addr_norm", "addr_null", "addr_nums"]


def _chunk(args):
    names, addrs = args
    load_dicts()
    rows = [norm_record(n, a) for n, a in zip(names, addrs)]
    return list(zip(*rows)) if rows else [[] for _ in COLS]


SCHEMA = {c: (pl.Boolean if c in ("nonlatin", "web", "alias", "idtag", "addr_null") else pl.Utf8) for c in COLS}


def normalize_frame(df: pl.DataFrame, n_jobs=None, chunk=50_000, slice_rows=1_000_000) -> pl.DataFrame:
    """Normalize in 1M-row slices so only one slice is ever held as Python objects."""
    outs = []
    with Pool(n_jobs or C.N_JOBS) as p:
        for off in range(0, df.height, slice_rows):
            sl = df.slice(off, slice_rows)
            names, addrs = sl["business_name"].to_list(), sl["business_address"].to_list()
            tasks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
            res = p.map(_chunk, tasks)
            outs.append(pl.DataFrame({c: [v for r in res for v in r[j]] for j, c in enumerate(COLS)}, schema=SCHEMA))
            del names, addrs, tasks, res
    return pl.concat([df.select("entity_id", "country"), pl.concat(outs)], how="horizontal")


def normalize_all(splits=C.SPLITS):
    C.NORM_DIR.mkdir(parents=True, exist_ok=True)
    nd, ad = load_dicts()
    print(f"dicts: name {nd} addr {ad} entries; USE_HAND_MAPS={C.USE_HAND_MAPS}")
    for split in splits:
        for src in (1, 2, 3):
            df = io.read_source(split, src)
            out = normalize_frame(df)
            for c in ("name_norm", "addr_norm"):
                assert out[c].null_count() == 0
            out.write_parquet(io.norm_path(split, src))
            stats = out.group_by("country").agg(
                pl.col("nonlatin").mean().alias("nonlatin"), pl.col("web").mean().alias("web"),
                pl.col("alias").mean().alias("alias"), pl.col("addr_null").mean().alias("addr_null"),
                (pl.col("name_core") == "").mean().alias("empty_name")).sort("country")
            print(f"{split} s{src}: {out.height:,} rows")
            with pl.Config(float_precision=4):
                print(stats)
