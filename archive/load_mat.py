import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


# -------- MATLAB → Python Conversion -------- #


def is_mat_struct(x):
    return getattr(x, "__class__").__name__ == "mat_struct"


def todict(obj):
    """Convert MATLAB struct to Python dict recursively."""
    out = {}
    for field in getattr(obj, "_fieldnames", []):
        out[field] = convert(getattr(obj, field))
    return out


def convert(x):
    """Recursively convert MATLAB-loaded objects into Python types."""
    if is_mat_struct(x):
        return todict(x)

    if isinstance(x, np.ndarray):
        if x.dtype == object:  # cell arrays, struct arrays
            return [convert(el) for el in x]
        return x  # numeric ndarrays → keep

    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()

    return x  # lists, dicts, scalars, strings


# -------- DataFrame -------- #


def to_dataframe(val):
    """Convert the converted Python object into a DataFrame."""
    # dict → columns
    if isinstance(val, dict):
        try:
            return pd.DataFrame({k: to_series(v) for k, v in val.items()})
        except Exception:
            return pd.DataFrame([{k: str(v) for k, v in val.items()}])

    # list → rows or single column
    if isinstance(val, list):
        if all(isinstance(el, dict) for el in val):
            return pd.DataFrame(val)
        return pd.DataFrame(val)

    # ndarray → let pandas handle 1D or 2D
    if isinstance(val, np.ndarray):
        return pd.DataFrame(val)

    # scalar → single-row DataFrame
    return pd.DataFrame([val])


def to_series(v):
    """Normalize values so DataFrame constructor accepts them."""
    if isinstance(v, np.ndarray):
        return pd.Series(v) if v.ndim == 1 else v.tolist()
    if isinstance(v, list):
        return pd.Series(v)
    return v


# -------- Save -------- #


def save_mat_vars_to_csv(mat_path: str, out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)

    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    for key, raw_val in mat.items():
        if key.startswith("__"):
            continue  # skip metadata

        converted = convert(raw_val)
        df = to_dataframe(converted)
        out_path = os.path.join(out_dir, f"{key}.csv")

        try:
            df.to_csv(out_path, index=False)
            print(f"Saved {key:<12} → {out_path}")
        except Exception as e:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(str(converted))
            print(f"Fallback saved {key} as text → {out_path} ({e})")


# -------- Script Entry -------- #

if __name__ == "__main__":
    mat_file = os.path.join("data", "s2", "S2_A1_E2.mat")
    save_mat_vars_to_csv(mat_file)
