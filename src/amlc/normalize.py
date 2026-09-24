"""Name / address normalisation (vectorised with polars; ~1-2 min per million rows).

Adds columns to a source frame:
    name_norm   transliterated, lower-case, punctuation-free, abbreviations canonicalised,
                phone numbers / web suffixes removed
    name_core   name_norm without legal-form tokens (inc, llc, pvt, ltd, sarl, ...)
    legal       the legal-form tokens that were removed (sorted, space-joined)
    addr_norm   same treatment for the address, with street types and state names canonicalised
    addr_nums   list of numeric tokens in the address (house / plot / unit numbers)
    non_latin   the raw name or address used a non-Latin script (Devanagari, Tamil, Kannada, ...)

Nothing here is country-specific in a way that breaks on unseen countries (France is test-only):
the maps only fire when a token matches, everything else passes through.
"""
import string

import polars as pl
from unidecode import unidecode

from .config import ADDR_COL, NAME_COL


def _collapse(s: str) -> str:
    """'limittedd' -> 'limited'. Mirrors the doubled-letter collapse applied to the data."""
    out = []
    for ch in s:
        if not (out and ch == out[-1] and ch.isalpha()):
            out.append(ch)
    return "".join(out)


# Legal forms (US / India / France + common transliterations). Keys are collapsed forms.
_LEGAL = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "lc": "llc", "ltd": "ltd", "limited": "ltd", "limted": "ltd", "lp": "lp", "llp": "llp",
    "pllc": "pllc", "pc": "pc", "plc": "plc", "pvt": "pvt", "private": "pvt", "priv": "pvt",
    "praivet": "pvt", "praivt": "pvt", "prayvet": "pvt", "praivett": "pvt", "piraivet": "pvt",
    "limitedd": "ltd", "limitted": "ltd", "limitet": "ltd", "limitedu": "ltd",
    "opc": "opc", "elelpi": "llp", "elpi": "llp", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
    "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc", "scp": "scp", "selarl": "selarl",
    "gmbh": "gmbh", "dba": "dba",
}
_LEGAL = {_collapse(k): v for k, v in _LEGAL.items()}
# multi-letter abbreviations written with dots/spaces collapse to these after punctuation removal
_NAME_WORDS = {"and": "and", "et": "and", "n": "and", "intl": "international", "svcs": "services",
               "svc": "service", "mfg": "manufacturing", "assoc": "associates", "bros": "brothers",
               "dept": "department", "ctr": "center", "centre": "center", "tech": "technologies"}
_NAME_DROP = {"com", "www", "net", "org", "in", "the", "le", "la", "les", "de", "du", "des"}

_ADDR_WORDS = {
    # US / generic street types
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr", "lane": "ln",
    "boulevard": "blvd", "bd": "blvd", "court": "ct", "place": "pl", "circle": "cir", "highway": "hwy",
    "parkway": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq", "suite": "ste", "apartment": "apt",
    "unit": "unit", "building": "bldg", "floor": "fl", "north": "n", "south": "s", "east": "e", "west": "w",
    "mount": "mt", "saint": "st", "sainte": "ste", "fort": "ft", "number": "no", "num": "no",
    # India
    "nagar": "ngr", "colony": "col", "sector": "sec", "marg": "marg", "opposite": "opp", "near": "nr",
    "behind": "bhd", "main": "main", "cross": "cross", "block": "blk", "plot": "plot", "door": "door",
    # France
    "chemin": "ch", "impasse": "imp", "allee": "all", "quai": "quai", "route": "rte",
    # noise
    "cdp": "", "po": "po", "box": "box",
}
_STATES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    # India (single-token names; multi-word ones handled in _PHRASES)
    "maharashtra": "mh", "karnataka": "ka", "kerala": "kl", "gujarat": "gj", "rajasthan": "rj",
    "delhi": "dl", "telangana": "tg", "bihar": "br", "odisha": "od", "orissa": "od", "punjab": "pb",
    "haryana": "hr", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "goa": "ga",
}
# multi-word replacements applied on the joined string before tokenising
_PHRASES = {
    "new york": "ny", "new jersey": "nj", "new mexico": "nm", "new hampshire": "nh", "north carolina": "nc",
    "south carolina": "sc", "north dakota": "nd", "south dakota": "sd", "west virginia": "wv",
    "rhode island": "ri", "district of columbia": "dc", "tamil nadu": "tn", "uttar pradesh": "up",
    "madhya pradesh": "mp", "andhra pradesh": "ap", "west bengal": "wb", "himachal pradesh": "hp",
    "jammu and kashmir": "jk",
}
_NAME_PHRASES = {"private limited": "pvt ltd", "limited liability company": "llc", "et cie": "and co"}
_ADDR_WORDS = {_collapse(k): v for k, v in {**_ADDR_WORDS, **_STATES}.items()}
_NAME_WORDS = {_collapse(k): v for k, v in _NAME_WORDS.items()}

_DOUBLES = [c * 2 for c in string.ascii_lowercase]
_SINGLES = list(string.ascii_lowercase)


def _translit(col: pl.Expr) -> pl.Expr:
    """unidecode only the rows that need it (non-ASCII), the rest stays vectorised."""
    return (
        pl.when(col.str.contains(r"[^\x00-\x7F]"))
        .then(col.map_elements(unidecode, return_dtype=pl.Utf8, skip_nulls=True))
        .otherwise(col)
    )


def _base_clean(col: pl.Expr, phrases: dict) -> pl.Expr:
    e = _translit(col.fill_null("")).str.to_lowercase()
    e = e.str.replace_all("&", " and ").str.replace_all(r"[^a-z0-9 ]+", " ")
    for _ in range(2):  # 'aaa' needs two passes
        e = e.str.replace_many(_DOUBLES, _SINGLES)
    e = e.str.replace_all(r"\s+", " ").str.strip_chars()
    if phrases:
        e = e.str.replace_many({_collapse(k): v for k, v in phrases.items()})
    return e


def _map_tokens(col: pl.Expr, mapping: dict, drop: set | None = None) -> pl.Expr:
    toks = col.str.split(" ")
    toks = toks.list.eval(pl.element().replace(mapping))
    if drop:
        toks = toks.list.eval(pl.element().filter(~pl.element().is_in(list(drop))))
    return toks.list.eval(pl.element().filter(pl.element() != "")).list.join(" ")


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    name = pl.col(NAME_COL)
    addr = pl.col(ADDR_COL)
    non_latin = r"[^\x00-\x{24F}]"  # anything beyond Latin / Latin-extended (accents are fine)
    out = df.with_columns(
        non_latin=name.fill_null("").str.contains(non_latin) | addr.fill_null("").str.contains(non_latin),
        _name=_base_clean(name, _NAME_PHRASES).str.replace_all(r"\b\d{7,}\b", " "),
        _addr=_base_clean(addr, _PHRASES),
    )
    out = out.with_columns(
        name_norm=_map_tokens(pl.col("_name"), {**_NAME_WORDS, **_LEGAL}, drop=_NAME_DROP),
        addr_norm=_map_tokens(pl.col("_addr"), _ADDR_WORDS),
    )
    toks = pl.col("name_norm").str.split(" ")
    legal_vals = sorted(set(_LEGAL.values()))
    out = out.with_columns(
        name_core=toks.list.eval(pl.element().filter(~pl.element().is_in(legal_vals))).list.join(" "),
        legal=toks.list.eval(pl.element().filter(pl.element().is_in(legal_vals))).list.sort().list.join(" "),
        addr_nums=pl.col("addr_norm").str.extract_all(r"\b\d+\b")
        .list.eval(pl.element().str.strip_chars_start("0").replace("", "0")),
    )
    # a name that is only legal tokens ("LLC") keeps its full form as core
    out = out.with_columns(
        name_core=pl.when(pl.col("name_core") == "").then(pl.col("name_norm")).otherwise(pl.col("name_core"))
    )
    return out.drop("_name", "_addr")
