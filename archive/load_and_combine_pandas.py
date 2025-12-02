import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# --- Variables we want to extract & combine (same logic as before) --- #
VAR_NAMES = [
    "emg",
    "glove",
    "repetition",
    "rerepetition",
    "stimulus",
    "restimulus",
]


# --- MATLAB → Python conversion helpers (from your old load_mat.py logic) --- #

def is_mat_struct(x):
    """Check if an object is a MATLAB struct as loaded by scipy.io.loadmat."""
    return getattr(x, "__class__", None).__name__ == "mat_struct"


def todict(obj):
    """Convert MATLAB struct to Python dict recursively."""
    out = {}
    for field in getattr(obj, "_fieldnames", []):
        out[field] = convert(getattr(obj, field))
    return out


def convert(x):
    """Recursively convert MATLAB-loaded objects into Python-native types."""
    if is_mat_struct(x):
        return todict(x)

    if isinstance(x, np.ndarray):
        if x.dtype == object:  # cell arrays / struct arrays
            return [convert(el) for el in x]
        return x  # numeric ndarray

    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()

    # lists, dicts, scalars, strings → leave as-is
    return x


# --- Conversion to DataFrame --- #

def to_dataframe(val):
    """Convert a converted Python object into a pandas DataFrame."""
    # dict → columns
    if isinstance(val, dict):
        try:
            return pd.DataFrame({k: to_series(v) for k, v in val.items()})
        except Exception:
            # fallback: one row with stringified values
            return pd.DataFrame([{k: str(v) for k, v in val.items()}])

    # list → rows or single column
    if isinstance(val, list):
        if all(isinstance(el, dict) for el in val):
            return pd.DataFrame(val)
        return pd.DataFrame(val)

    # ndarray → 1D or 2D (let pandas handle it)
    if isinstance(val, np.ndarray):
        return pd.DataFrame(val)

    # scalar → single-row DataFrame
    return pd.DataFrame([val])


def to_series(v):
    """Normalize values so DataFrame constructor accepts them as columns."""
    if isinstance(v, np.ndarray):
        return pd.Series(v) if v.ndim == 1 else v.tolist()
    if isinstance(v, list):
        return pd.Series(v)
    return v


def _prefix_columns(df: pd.DataFrame, stem: str) -> pd.DataFrame:
    """Prefix DataFrame column names with the given stem."""
    if df.shape[1] == 0:
        return df
    return df.rename(columns={c: f"{stem}_{c}" for c in df.columns})


# --- Core function: load one .mat and return combined DataFrame --- #

def load_and_combine_single(mat_path: str) -> pd.DataFrame:
    """
    Load a single Ninapro .mat file and combine selected variables
    (emg, glove, repetition, rerepetition, stimulus, restimulus)
    into one pandas DataFrame.
    """
    mat_path = Path(mat_path)
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    dfs = []

    for var in VAR_NAMES:
        if var not in mat:
            print(f"[{mat_path.name}] Warning: variable '{var}' not found, skipping.")
            continue

        raw_val = mat[var]
        converted = convert(raw_val)
        df = to_dataframe(converted)
        df_prefixed = _prefix_columns(df, var)
        dfs.append(df_prefixed)
        print(f"[{mat_path.name}] Loaded '{var}' with shape {df.shape}")

    if not dfs:
        raise ValueError(f"No expected variables found in {mat_path}")

    combined = pd.concat(dfs, axis=1)
    print(f"[{mat_path.name}] Combined DataFrame shape: {combined.shape}")
    return combined


# --- Save helper: same name as .mat, but .parquet --- #

def save_combined_df(df: pd.DataFrame, mat_path: str):
    mat_path = Path(mat_path)
    out_path = mat_path.with_suffix(".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{mat_path.name}] Saved combined data → {out_path}")


# --- Main entry: handle multiple .mat paths --- #

def process_mat_files(mat_files):
    """
    Process one or multiple .mat files.
    For each, load & combine, then save as a .parquet file
    with the same path and basename as the .mat file.
    """
    if isinstance(mat_files, (str, Path)):
        mat_files = [mat_files]

    for mat_file in mat_files:
        df = load_and_combine_single(mat_file)
        save_combined_df(df, mat_file)


if __name__ == "__main__":

    # If no arguments: use multiple default files
    if len(sys.argv) <= 1:
        default_mats = [
            os.path.join("data", "s2", "S2_A1_E1.mat"),
            # os.path.join("data", "s2", "S2_A1_E2.mat"),
            # os.path.join("data", "s2", "S2_A1_E3.mat"),
        ]
        print("No .mat paths provided. Using defaults:")
        for m in default_mats:
            print("  -", m)

        process_mat_files(default_mats)

    else:
        # Use paths provided as CLI arguments
        process_mat_files(sys.argv[1:])

