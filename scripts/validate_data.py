"""
Validate a CSV of transactions against the Pandera schema.

Why:
- Enforces the data contract before any training or inference.
- Catches wrong types, bad dates, out-of-range values, and invalid categories.
- Provides clear, actionable error messages for debugging.

Usage:
    python scripts/validate_data.py data/transactions.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from src.validation.schema import TransactionsSchema


def main(path: str) -> None:
    """Validate the given CSV and print a success message or raise on failure.

    Behavior:
        1) Reads the CSV into a DataFrame.
        2) Applies a pre-rule: drops rows where amount == 0 (not useful for training).
        3) Runs Pandera validation in 'lazy' mode to collect all failures at once.

    Args:
        path: Filepath to a CSV with columns matching the data contract.

    Raises:
        pandera.errors.SchemaError if validation fails.
    """
    df = pd.read_csv(path)

    # Pre-rule from README: ignore zero-amount rows.
    if "amount" in df.columns:
        df = df[df["amount"].astype(float) > 0].copy()

    # Validate against the schema; lazy=True aggregates all errors.
    TransactionsSchema.validate(df, lazy=True)

    print(f"Validation OK: {len(df)} rows in {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_data.py data/transactions.csv")
        sys.exit(1)
    main(sys.argv[1])
