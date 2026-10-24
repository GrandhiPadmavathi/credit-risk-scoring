"""Load the UCI 'default of credit card clients' dataset into DuckDB with clean column names."""

from pathlib import Path

import duckdb
import pandas as pd

from creditrisk.config import SETTINGS

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
TARGET = "defaulted"

RAW_COLUMNS = (
    ["id", "limit_bal", "sex", "education", "marriage", "age"]
    + [f"pay_{m}" for m in range(1, 7)]
    + [f"bill_amt{m}" for m in range(1, 7)]
    + [f"pay_amt{m}" for m in range(1, 7)]
)


def read_raw(path: Path = SETTINGS.paths.raw_xls) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df.columns = [c.lower() for c in df.columns]
    # The source skips PAY_1; PAY_0 is the most recent month (September 2005).
    df = df.rename(columns={"pay_0": "pay_1", "default payment next month": TARGET})
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented category codes into 'other', as recommended in the dataset notes."""
    df = df.copy()
    df["education"] = df["education"].where(df["education"].isin([1, 2, 3]), 4)
    df["marriage"] = df["marriage"].where(df["marriage"].isin([1, 2]), 3)
    return df


def write_db(df: pd.DataFrame, db_path: Path = SETTINGS.paths.db) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.register("src", df)
        con.execute("CREATE OR REPLACE TABLE applicants AS SELECT * FROM src")


def load_applicants(db_path: Path = SETTINGS.paths.db) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute("SELECT * FROM applicants ORDER BY id").df()
