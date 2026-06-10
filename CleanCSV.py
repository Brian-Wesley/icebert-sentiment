#!/usr/bin/env python3
"""
Fix Icelandic CSV extraction artifacts *without* producing mojibake (garbled characters).

Key fixes:
- Read as UTF-8 with BOM (utf-8-sig)
- Remove soft hyphen (U+00AD) + other invisible separators
- Normalize Unicode (NFC) + whitespace
- If text is already mojibake (e.g., "FjÃ³rÃ°a"), try a safe undo
- Write as UTF-8 with BOM (utf-8-sig) so Excel opens it correctly

INPUT (your uploaded file):
  /mnt/data/icelandic_sentiment_v1.4.csv

OUTPUT:
  /mnt/data/icelandic_sentiment_v1.4.clean.csv
"""

import re
import unicodedata
import pandas as pd

INPUT_PATH = "./icelandic_sentiment_v1.1.csv"
OUTPUT_PATH = "./icelandic_sentiment_v1.1.clean.csv"

INVISIBLE = re.compile(r"[\u00ad\u200b\u200c\u200d\ufeff\u2060]")  # soft hyphen + zero-width + BOM + word joiner
WS = re.compile(r"\s+")

def try_unmojibake(s: str) -> str:
    """
    If a string looks like UTF-8 that was incorrectly decoded as Latin-1/CP1252,
    try to reverse it. Only triggers on common mojibake markers.
    """
    if not s:
        return s
    # Typical markers when UTF-8 got mis-decoded
    if ("Ã" in s) or ("Â" in s) or ("Ð" in s) or ("ðŸ" in s):
        for enc in ("latin1", "cp1252"):
            try:
                fixed = s.encode(enc, errors="strict").decode("utf-8", errors="strict")
                return fixed
            except Exception:
                pass
    return s

def clean_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)

    # First: undo mojibake if it exists
    s = try_unmojibake(s)

    # Then: normalize + remove extraction artifacts
    s = unicodedata.normalize("NFC", s)
    s = INVISIBLE.sub("", s)
    s = s.replace("\u00a0", " ")          # NBSP -> normal space
    s = WS.sub(" ", s).strip()            # collapse whitespace
    return s

def pick_text_column(df: pd.DataFrame) -> str:
    if "text" in df.columns:
        return "text"
    for c in df.columns:
        if str(c).lower() in {"label", "sentiment", "target", "y"}:
            continue
        if df[c].dtype == "object":
            return c
    raise ValueError("Couldn't infer the text column. Rename it to 'text' or edit pick_text_column().")

def main():
    # Read explicitly as UTF-8 with BOM
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # Drop accidental columns (your file has trailing commas)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed:\s*\d+$")]

    text_col = pick_text_column(df)
    df[text_col] = df[text_col].map(clean_text)

    # Drop rows that become empty after cleaning (optional)
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)

    # Write with UTF-8 BOM so Excel won’t show garbled Icelandic
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print("\nPreview (first 5 cleaned texts):")
    print(df[text_col].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
