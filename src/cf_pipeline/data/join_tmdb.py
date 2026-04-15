import pandas as pd


def join_movies_with_tmdb(
    movies: pd.DataFrame,
    links: pd.DataFrame,
    tmdb: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join movies → links → TMDB metadata; keeps only fully matched items.

    Args:
        movies: ML-1M movies DataFrame with columns [item_id, title, genres].
        links:  Mapping DataFrame with columns [item_id, imdb_id].
                imdb_id must already be in 'tt0XXXXXXX' string format.
        tmdb:   TMDB metadata DataFrame with column imdb_id plus any metadata
                columns (overview, popularity, vote_average, etc.).
                Must already be deduplicated on imdb_id.

    Returns:
        DataFrame with all movies columns + all tmdb columns for matched items.
        Items with no link or no TMDB match are silently dropped (inner join).
        Index is reset to 0-based range.
    """
    # Step 1: attach imdb_id to each movie (inner → drops movies with no link)
    m = movies.merge(links[["item_id", "imdb_id"]], on="item_id", how="inner")

    # Step 2: attach TMDB metadata (inner → drops movies not in TMDB)
    # Both DataFrames share 'title' and 'genres'; suffix to avoid ambiguity,
    # then rename to descriptive names for all downstream consumers.
    out = m.merge(tmdb, on="imdb_id", how="inner", suffixes=("", "_tmdb"))

    # Rename to unambiguous names:
    #   title        → ML-1M title string e.g. "Toy Story (1995)"
    #   genres       → ML-1M pipe-separated string e.g. "Animation|Children's|Comedy"
    #   tmdb_title   → TMDB title without year  e.g. "Toy Story"
    #   tmdb_genres  → TMDB JSON genre list for LLM stage
    rename = {}
    if "title_tmdb"  in out.columns: rename["title_tmdb"]  = "tmdb_title"
    if "genres_tmdb" in out.columns: rename["genres_tmdb"] = "tmdb_genres"
    if rename:
        out = out.rename(columns=rename)

    return out.reset_index(drop=True)
