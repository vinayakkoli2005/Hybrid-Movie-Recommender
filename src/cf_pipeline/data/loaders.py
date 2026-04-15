from pathlib import Path
import pandas as pd


def load_ml1m_ratings(raw_dir: str | Path) -> pd.DataFrame:
    path = Path(raw_dir) / "ratings.dat"
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "item_id": "int32", "rating": "int8", "timestamp": "int64"},
    )
    return df


def load_ml1m_movies(raw_dir: str | Path) -> pd.DataFrame:
    path = Path(raw_dir) / "movies.dat"
    df = pd.read_csv(
        path, sep="::", engine="python", header=None, encoding="latin-1",
        names=["item_id", "title", "genres"],
        dtype={"item_id": "int32"},
    )
    return df


def load_links(raw_dir: str | Path) -> pd.DataFrame:
    """Load ML-1M → IMDb ID mapping from links.csv.

    The raw links.csv (from ML-25M) stores imdbId as a plain integer (e.g. 114709).
    This function formats it to the canonical 'tt0XXXXXXX' string expected by
    join_movies_with_tmdb and TMDB lookups.

    Returns:
        DataFrame with columns [item_id, imdb_id] where imdb_id is 'tt0XXXXXXX'.
    """
    path = Path(raw_dir) / "links.csv"
    df = pd.read_csv(path, dtype={"movieId": "int32", "imdbId": str})
    df = df.rename(columns={"movieId": "item_id"})
    # Zero-pad to 7 digits and prepend 'tt' → matches TMDB imdb_id format
    df["imdb_id"] = "tt" + df["imdbId"].str.strip().str.zfill(7)
    return df[["item_id", "imdb_id"]].reset_index(drop=True)


def load_tmdb_metadata(raw_dir: str | Path) -> pd.DataFrame:
    """Load and clean TMDB movies_metadata.csv.

    Cleaning steps applied:
    1. Keep only rows where imdb_id starts with 'tt' (drops ~20 malformed rows).
    2. Deduplicate on imdb_id — keep the row with the highest vote_count
       (the dataset has ~30 duplicate imdb_ids).
    3. Retain only the columns useful for this project.

    Returns:
        DataFrame keyed by imdb_id with TMDB metadata columns.
    """
    path = Path(raw_dir) / "movies_metadata.csv"
    df = pd.read_csv(path, low_memory=False)

    # Drop rows with missing or malformed imdb_id
    df = df[df["imdb_id"].astype(str).str.startswith("tt")].copy()

    # Deduplicate: keep highest-voted row per imdb_id
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    df = df.sort_values("vote_count", ascending=False).drop_duplicates("imdb_id", keep="first")

    keep = ["imdb_id", "title", "overview", "genres", "popularity",
            "vote_average", "vote_count", "release_date", "runtime", "tagline"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].reset_index(drop=True)
