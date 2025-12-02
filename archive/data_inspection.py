import pandas as pd


def create_series_id(csv_path: str):
    """
    Create a Series identifying each series of the csv inplace.
    Using the value in the stimulation, restimulation, repetition and rerepetition columns.
    """
    df = pd.read_csv(csv_path)
    # iterate over the rows and create a new column 'series_id'
    df["series_id"] = (
        df["stimulus_0"].astype(str)
        + "_"
        + df["restimulus_0"].astype(str)
        + "_"
        + df["repetition_0"].astype(str)
        + "_"
        + df["rerepetition_0"].astype(str)
    )
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    create_series_id("data/data.csv")
