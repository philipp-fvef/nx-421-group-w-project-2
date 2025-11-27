import os
from pathlib import Path
import pandas as pd

FILES = [
    "emg.csv",
    "glove.csv",
    "repetition.csv",
    "rerepetition.csv",
    "stimulus.csv",
    "restimulus.csv",
]


def _read_and_prefix(path: Path) -> pd.DataFrame:
    """Read CSV, drop accidental index columns, and prefix columns with file stem."""
    df = pd.read_csv(path)

    # If no columns remain (empty), return empty DataFrame
    if df.shape[1] == 0:
        return df
    stem = path.stem

    # Prefix column names to avoid collisions
    df = df.rename(columns={c: f"{stem}_{c}" for c in df.columns})
    return df


def combine_csvs(data_dir: str = "data", out_file: str = "data/data.csv"):
    data_path = Path(data_dir)
    dfs = []
    for fname in FILES:
        p = data_path / fname
        if not p.exists():
            print(f"Warning: {p} not found, skipping.")
            continue
        try:
            df = _read_and_prefix(p)
            dfs.append(df)
            print(f"Loaded {p} (shape={df.shape})")
        except Exception as e:
            print(f"Error reading {p}: {e}")

    if not dfs:
        print("No files loaded. Nothing to combine.")
        return

    # Concatenate side-by-side, aligning on index (rows). Missing values become NaN.
    combined = pd.concat(dfs, axis=1)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Combined CSV written to {out_path} (shape={combined.shape})")


if __name__ == "__main__":
    combine_csvs()
