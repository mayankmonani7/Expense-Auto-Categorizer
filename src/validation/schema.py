"""
Pandera schema that encodes the data contract for transactions.

This is the single source of truth for:
- Column names and types
- Allowed categories/modes
- Basic semantic checks (date format, amount >= 0, description length)

Why Pandera?
- Lightweight, runs in CI, and gives precise failure messages (great for debugging).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Set

import pandera as pa
from pandera import Check, Column, DataFrameSchema

# ---- Enumerations used across the project ----
CATEGORIES: List[str] = [
    "Food",
    "Transport",
    "Bills",
    "Groceries",
    "Shopping",
    "Rent",
    "Health",
    "Insurance",
    "Travel",
    "Entertainment",
]
MODES: Set[str] = {"UPI", "Card", "NetBanking", "NEFT", "Cash", "Other"}


def _is_iso_date(s: str) -> bool:
    """Return True if a string matches the strict 'YYYY-MM-DD' format.

    We use a function (rather than regex) to catch impossible dates like
    2025-02-30, which `datetime.strptime` will reject.

    Args:
        s: Candidate date string.

    Returns:
        bool: Whether the string is a valid ISO calendar date.
    """
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except Exception:
        return False


# ---- DataFrame schema (data contract) ----
TransactionsSchema: DataFrameSchema = DataFrameSchema(
    {
        # ID must exist; uniqueness is enforced upstream (generator/ingestion).
        "id": Column(pa.String, nullable=False, coerce=True),
        # Strict ISO date format; use `.map` so we validate elementwise.
        "date": Column(
            pa.String, checks=Check(lambda s: s.map(_is_iso_date).all()), nullable=False
        ),
        # Keep descriptions usable for NLP: non-empty and bounded.
        "description": Column(pa.String, checks=Check.str_length(3, 300), nullable=False),
        # Merchant can be empty string but must be a string.
        "merchant": Column(pa.String, nullable=True, coerce=True),
        # Amounts are non-negative; coerce to float to catch textual inputs early.
        "amount": Column(pa.Float, checks=Check.ge(0), nullable=False, coerce=True),
        # Mode is optional at training time but must be from the known set when present.
        "mode": Column(pa.String, checks=Check.isin(list(MODES)), nullable=True),
        # Label is required for training rows; for inference it can be missing/ignored.
        "label": Column(pa.String, checks=Check.isin(CATEGORIES), nullable=True),
    },
    # Reject unexpected columns to avoid silent schema drift.
    strict=True,
    # Coerce types on read to reduce downstream surprises.
    coerce=True,
)
