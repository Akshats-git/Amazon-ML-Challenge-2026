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

# R08 France-only rules (C.FR_NORM_ADDR / C.FR_NORM_NAME; France has no train rows, so the models, pools and OOF are unaffected). S2/S3
# abbreviate street types and add the department where S1 spells them out and ends with the region.
# Targets are the exact tokens S1 uses after the doubled-letter collapse, mined from confident test France top-1 pairs
# (work/logs/diag_fr_maps.log: q token -> S1 token, purity >= 0.97).
FR_ADDR_MAP = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "blvd": "boulevard", "pl": "place",
    "ch": "chemin", "chem": "chemin", "imp": "impase", "rte": "route", "al": "ale", "crs": "cours", "q": "quai",
    "res": "residence", "psg": "pasage", "pas": "pasage", "apt": "apartement", "ap": "apartement", "st": "saint",
    "ste": "sainte",
}
FR_ADDR_AFTER_NUM = {"b": "bis", "t": "ter"}  # 12 b -> 12 bis
FR_NUM_PREFIX = {"no", "ndeg"}  # "No 12", "N° 12" (unidecode: ndeg); S1 writes the bare number
RE_NDEG = re.compile(r"ndeg(\d+)")
# regions (S1) and departments (S2/S3, at any position) are dropped on both sides; S1 has nord/gironde in < 200 rows
FR_REGIONS = [tuple(p.split()) for p in ("hauts de france", "nouvele aquitaine", "pays de la loire", "loire atlantique",
                                         "pas de calais", "gironde", "nord")]
FR_NAME_MAP = {
    "et": "and", "compagnie": "cie", "frs": "freres", "st": "saint", "center": "centre", "cb": "club", "clb": "club",
    "svc": "service", "farmacie": "pharmacie",
}


def fr_name(toks: list) -> list:
    """Join runs of >= 2 single letters (s a r l -> sarl), then word map."""
    out, run = [], []
    for t in toks + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.extend(["".join(run)] if len(run) >= 2 else run)
            run = []
        if t:
            out.append(FR_NAME_MAP.get(t, t))
    return out


def fr_addr(toks: list) -> list:
    """Drop region/department phrases and the number prefix, expand abbreviations, strip leading zeros."""
    out, i = [], 0
    while i < len(toks):
        for ph in FR_REGIONS:
            if tuple(toks[i:i + len(ph)]) == ph:
                i += len(ph)
                break
        else:
            out.append(toks[i])
            i += 1
    res = []
    for j, t in enumerate(out):
        m = RE_NDEG.fullmatch(t)
        if m:
            t = m.group(1)
        elif t in FR_NUM_PREFIX and j + 1 < len(out) and out[j + 1][:1].isdigit():
            continue
        elif t in FR_ADDR_AFTER_NUM and res and res[-1].isdigit():
            t = FR_ADDR_AFTER_NUM[t]
        else:
            t = FR_ADDR_MAP.get(t, t)
        res.append((t.lstrip("0") or "0") if t.isdigit() else t)
    return res


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


def norm_addr(addr: str, nonlatin: bool, fr: bool):
    """Address part of norm_record -> (addr_norm, addr_null, addr_nums). fr: France address rules (fr_addr)."""
    addr_null = addr.strip().lower() in ADDR_NULL
    atoks = [] if addr_null else addr_tokens_pre(addr)
    if fr:
        atoks = fr_addr(atoks)
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
    return addr_norm, addr_null or not addr_norm, " ".join(nums)


def is_nonlatin(name: str, addr: str) -> bool:
    return bool(RE_NONLATIN.search(name) or RE_NONLATIN.search(addr))


def norm_record(name: str, addr: str, country: str = ""):
    ntoks = (lambda s: fr_name(name_tokens_pre(s))) if C.FR_NORM_NAME and country == "France" else name_tokens_pre
    nonlatin = is_nonlatin(name, addr)
    web = bool(RE_WEB.search(name))
    alias = bool(RE_ALIAS.search(name))
    idtag = bool(RE_IDTAG.search(name))
    name_norm, name_core, name_sq, legal = _name_post(ntoks(name), nonlatin)
    alias_l = alias_r = ""
    if alias:
        m = RE_ALIAS.search(name)
        alias_l = _name_post(ntoks(name[: m.start()]), nonlatin)[1]
        alias_r = _name_post(ntoks(name[m.end():]), nonlatin)[1]
    addr_norm, addr_null, addr_nums = norm_addr(addr, nonlatin, C.FR_NORM_ADDR and country == "France")
    return (name_norm, name_core, name_sq, legal, alias_l, alias_r, nonlatin, web, alias, idtag,
            addr_norm, addr_null, addr_nums)


COLS = ["name_norm", "name_core", "name_sq", "legal", "alias_l", "alias_r", "nonlatin", "web", "alias", "idtag",
        "addr_norm", "addr_null", "addr_nums"]


def _chunk(args):
    names, addrs, countries = args
    load_dicts()
    rows = [norm_record(n, a, c) for n, a, c in zip(names, addrs, countries)]
    return list(zip(*rows)) if rows else [[] for _ in COLS]


SCHEMA = {c: (pl.Boolean if c in ("nonlatin", "web", "alias", "idtag", "addr_null") else pl.Utf8) for c in COLS}


def normalize_frame(df: pl.DataFrame, n_jobs=None, chunk=50_000, slice_rows=1_000_000) -> pl.DataFrame:
    """Normalize in 1M-row slices so only one slice is ever held as Python objects."""
    outs = []
    with Pool(n_jobs or C.N_JOBS) as p:
        for off in range(0, df.height, slice_rows):
            sl = df.slice(off, slice_rows)
            names, addrs = sl["business_name"].to_list(), sl["business_address"].to_list()
            countries = sl["country"].to_list()
            tasks = [(names[i:i + chunk], addrs[i:i + chunk], countries[i:i + chunk]) for i in range(0, len(names), chunk)]
            res = p.map(_chunk, tasks)
            outs.append(pl.DataFrame({c: [v for r in res for v in r[j]] for j, c in enumerate(COLS)}, schema=SCHEMA))
            del names, addrs, countries, tasks, res
    return pl.concat([df.select("entity_id", "country"), pl.concat(outs)], how="horizontal")


def normalize_all(splits=C.SPLITS):
    C.NORM_DIR.mkdir(parents=True, exist_ok=True)
    nd, ad = load_dicts()
    print(f"dicts: name {nd} addr {ad} entries; USE_HAND_MAPS={C.USE_HAND_MAPS} FR_NORM_ADDR={C.FR_NORM_ADDR} FR_NORM_NAME={C.FR_NORM_NAME}")
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
