"""
src/ingest.py — Phase 2
Loads SP500_Alpha_Dataset_Final.csv, cleans it, converts each row
into a rich text Document, chunks, and saves to disk.

Run from finsight/ root:
    python src/ingest.py
"""
import os, pickle, random
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# 1. LOAD & CLEAN CSV
# ─────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    print(f"\n Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f" Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")

    financial_cols = ["Net_Margin", "ROA", "Revenue_Growth_YoY", "Net_Income_Growth_YoY"]
    nlp_cols       = ["MDA_Flesch_Kincaid", "MDA_FinBERT_Sentiment"]

    before = len(df)
    df = df.dropna(subset=financial_cols, how="all")
    print(f" Dropped {before - len(df)} rows with no financial data")

    numeric_cols = financial_cols + nlp_cols
    for col in numeric_cols:
        df[col] = df.groupby("Ticker")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].round(4)

    df["Year"]   = df["Year"].astype(str)
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    print(f" Clean: {df.shape[0]} rows | {df['Ticker'].nunique()} companies | Years: {sorted(df['Year'].unique())}")
    return df


# ─────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────

def _fmt_pct(v):
    try:    return f"{float(v)*100:.2f}%"
    except: return "N/A"

def _fmt_sentiment(v):
    try:
        f = float(v)
        label = "Positive" if f > 0.15 else "Negative" if f < -0.15 else "Neutral"
        return f"{label} ({f:+.3f})"
    except: return "N/A"

def _fmt_readability(v):
    try:
        f = float(v)
        if   f >= 16: label = "Very Complex (post-graduate)"
        elif f >= 13: label = "Complex (college level)"
        elif f >= 10: label = "Moderate (high school)"
        else:         label = "Simple (clear)"
        return f"Grade {f:.1f} — {label}"
    except: return "N/A"


# ─────────────────────────────────────────────
# 3. CONVERT ROWS → DOCUMENTS
# ─────────────────────────────────────────────

def rows_to_documents(df: pd.DataFrame) -> list:
    print(f"\n Converting {len(df)} rows to Documents ...")
    docs = []

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        year   = row["Year"]
        rev_g  = float(row["Revenue_Growth_YoY"])
        ni_g   = float(row["Net_Income_Growth_YoY"])

        gc_text = (
            "WARNING: Auditors issued a Going Concern warning. "
            "Serious doubt about ability to continue operations."
            if int(row["Flag_Going_Concern"]) == 1
            else "No going concern warning issued."
        )

        content = f"""Company: {ticker} | Fiscal Year: {year}

--- Financial Performance ---
Net Profit Margin      : {_fmt_pct(row['Net_Margin'])}
Return on Assets (ROA) : {_fmt_pct(row['ROA'])}
Revenue Growth YoY     : {_fmt_pct(row['Revenue_Growth_YoY'])}
Net Income Growth YoY  : {_fmt_pct(row['Net_Income_Growth_YoY'])}

--- Executive Narrative (MD&A Analysis) ---
FinBERT Sentiment Score : {_fmt_sentiment(row['MDA_FinBERT_Sentiment'])}
Writing Complexity      : {_fmt_readability(row['MDA_Flesch_Kincaid'])}
Auditor Warning         : {gc_text}

--- Summary ---
In fiscal year {year}, {ticker} reported a net margin of {_fmt_pct(row['Net_Margin'])} \
and ROA of {_fmt_pct(row['ROA'])}. Revenue {"grew" if rev_g >= 0 else "declined"} \
by {abs(rev_g)*100:.2f}% YoY, while net income {"grew" if ni_g >= 0 else "declined"} \
by {abs(ni_g)*100:.2f}%. Management writing was at {_fmt_readability(row['MDA_Flesch_Kincaid'])} \
reading level with {_fmt_sentiment(row['MDA_FinBERT_Sentiment'])} executive tone."""

        docs.append(Document(
            page_content=content,
            metadata={
                "ticker"        : ticker,
                "year"          : year,
                "going_concern" : str(int(row["Flag_Going_Concern"])),
                "sentiment_raw" : str(row["MDA_FinBERT_Sentiment"]),
                "source"        : f"{ticker}_{year}",
                "filename"      : f"{ticker}_{year}_10K.txt",
                "chunk_id"      : f"{ticker}_{year}",
            }
        ))

    print(f" Created {len(docs)} documents")
    return docs


# ─────────────────────────────────────────────
# 4. CHUNK
# ─────────────────────────────────────────────

def chunk_documents(documents: list, chunk_size: int = 800, chunk_overlap: int = 64) -> list:
    """
    chunk_size=800 keeps each full company record (~670 chars) in one chunk.
    chunk_overlap=64 provides continuity for the rare records that exceed 800 chars.
    """
    print(f"\n Chunking {len(documents)} documents (size={chunk_size}, overlap={chunk_overlap}) ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = str(i)
    print(f" Created {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# 5. SAVE / LOAD
# ─────────────────────────────────────────────

def save_chunks(chunks: list, path: str = "data/chunks.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"\n Saved {len(chunks)} chunks → {path} ({os.path.getsize(path)/1024:.1f} KB)")

def load_chunks(path: str = "data/chunks.pkl") -> list:
    with open(path, "rb") as f:
        chunks = pickle.load(f)
    print(f" Loaded {len(chunks)} chunks from {path}")
    return chunks


# ─────────────────────────────────────────────
# 6. INSPECT & STATS
# ─────────────────────────────────────────────

def inspect_chunks(chunks: list, n: int = 3) -> None:
    print(f"\n{'─'*65}\n  CHUNK INSPECTION — {n} random samples\n{'─'*65}")
    for i, chunk in enumerate(random.sample(chunks, min(n, len(chunks)))):
        print(f"\n  [Sample {i+1}] {chunk.metadata.get('ticker')} {chunk.metadata.get('year')} | {len(chunk.page_content)} chars")
        print("  " + chunk.page_content.replace("\n", "\n  "))
        print(f"\n{'─'*65}")

def print_stats(df: pd.DataFrame, chunks: list) -> None:
    gc      = int(df["Flag_Going_Concern"].sum())
    neg     = (df["MDA_FinBERT_Sentiment"] < -0.15).sum()
    pos     = (df["MDA_FinBERT_Sentiment"] >  0.15).sum()
    neu     = len(df) - neg - pos
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"\n{'─'*65}\n  DATASET SUMMARY\n{'─'*65}")
    print(f"  Total records       : {len(df)}")
    print(f"  Unique companies    : {df['Ticker'].nunique()}")
    print(f"  Years covered       : {sorted(df['Year'].unique())}")
    print(f"  Total chunks        : {len(chunks)}")
    print(f"  Avg chunk length    : {avg_len} chars")
    print(f"  Going concern flags : {gc} companies")
    print(f"  Sentiment           : {pos} positive | {neu} neutral | {neg} negative")
    print(f"  Avg net margin      : {df['Net_Margin'].mean()*100:.2f}%")
    print(f"  Avg ROA             : {df['ROA'].mean()*100:.2f}%")
    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    CSV_PATH    = "data/SP500_Alpha_Dataset_Final.csv"
    CHUNKS_PATH = "data/chunks.pkl"

    df        = load_csv(CSV_PATH)
    documents = rows_to_documents(df)
    chunks    = chunk_documents(documents)   # chunk_size=800, chunk_overlap=64
    inspect_chunks(chunks, n=3)
    print_stats(df, chunks)
    save_chunks(chunks, CHUNKS_PATH)
    print(" Phase 2 complete. Run src/embeddings.py next.")