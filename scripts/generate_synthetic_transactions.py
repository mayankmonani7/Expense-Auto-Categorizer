"""
Synthetic transaction generator for the Expense Auto-Categorizer.

This script creates a realistic, labeled CSV that matches our data contract:
id, date (YYYY-MM-DD), description, merchant, amount, mode, label.

Design goals:
- Be reproducible (fixed random seed).
- Reflect Indian-context merchants/phrases and mild real-world “mess”.
- Keep distributions reasonable (e.g., Rent amounts >> Food).
"""

from __future__ import annotations

import csv
import hashlib
import random
from datetime import date, timedelta
from typing import Dict, Iterable, List, Tuple

# Reproducibility
random.seed(42)

# ---- Taxonomy / Lookups ----
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

MODES: List[str] = ["UPI", "Card", "NetBanking", "NEFT", "Cash", "Other"]

# merchant -> list of (canonical_merchant, [description phrases])
MERCHANTS: Dict[str, List[Tuple[str, List[str]]]] = {
    "Food": [
        ("Zomato", ["zomato order", "zomato delivery", "zomato pizza deal"]),
        ("Swiggy", ["swiggy biryani", "swiggy lunch", "swiggy dinner"]),
        ("Starbucks", ["starbucks cappuccino", "starbucks latte", "starbucks cold brew"]),
        ("Dominos", ["dominos pizza", "dominos order", "dominos combo"]),
    ],
    "Transport": [
        ("Uber", ["uber ride to office", "uber*trip #{}", "uber airport ride"]),
        ("Ola", ["ola cab city ride", "ola outstation", "ola*trip {}"]),
        ("Metro", ["metro smart card topup", "metro ticket", "metro pass renewal"]),
        ("IOCL", ["petrol pump iocl", "fuel iocl", "iocl fuel refill"]),
    ],
    "Bills": [
        ("TNEB", ["electricity tneb bill", "tneb online payment"]),
        ("BSNL", ["bsnl broadband bill", "bsnl fiber bill"]),
        ("Airtel", ["airtel postpaid bill", "airtel dth recharge"]),
        ("TWAD", ["water board bill", "twad water payment"]),
    ],
    "Groceries": [
        ("BigBasket", ["bigbasket veggies", "bigbasket grocery basket"]),
        ("Reliance Fresh", ["reliance fresh groceries", "reliance fresh milk & eggs"]),
        ("More", ["more supermarket staples", "more grocery bag"]),
        ("JioMart", ["jiomart monthly groceries", "jiomart dal & rice"]),
    ],
    "Shopping": [
        ("Amazon", ["amazon basics hdmi cable", "amazon t-shirt", "amazon shoes"]),
        ("Flipkart", ["flipkart tshirt", "flipkart headphones", "flipkart home decor"]),
        ("Myntra", ["myntra kurti", "myntra jeans", "myntra footwear"]),
        ("Croma", ["croma usb-c charger", "croma power bank"]),
    ],
    "Rent": [
        ("Bank Transfer", ["rent to landlord", "monthly house rent"]),
        ("GPay", ["rent via gpay", "gpay rent transfer"]),
    ],
    "Health": [
        ("Apollo Pharmacy", ["apollo pharmacy medicine", "apollo antibiotic"]),
        ("Healthkart", ["healthkart whey protein", "healthkart vitamins"]),
        ("Dr Lal PathLabs", ["diagnostics blood test", "pathlab full checkup"]),
        ("Fortis", ["hospital opd fees", "fortis consultation"]),
    ],
    "Insurance": [
        ("LIC", ["lic premium", "lic quarterly premium"]),
        ("HDFC Ergo", ["hdfc ergo motor insurance", "hdfc ergo health"]),
        ("ICICI Lombard", ["icici lombard renewal", "lombard policy premium"]),
    ],
    "Travel": [
        ("IRCTC", ["irctc train booking", "irctc tatkal ticket"]),
        ("IndiGo", ["indigo flight add-on", "indigo web check-in"]),
        ("Goibibo", ["goibibo hotel booking", "goibibo bus ticket"]),
        ("OYO", ["oyo hotel stay", "oyo room booking"]),
    ],
    "Entertainment": [
        ("Netflix", ["netflix monthly", "netflix subscription"]),
        ("Spotify", ["spotify subscription", "spotify family plan"]),
        ("BookMyShow", ["movie tickets bookmyshow", "bms weekend show"]),
        ("SonyLIV", ["sonyliv annual", "sonyliv quarterly"]),
    ],
}


def random_date(start: date = date(2025, 7, 1), end: date = date(2025, 8, 10)) -> date:
    """Return a random date within [start, end] (inclusive).

    Why:
        We want timestamps to look realistic and support time-based splits later.

    Args:
        start: Earliest allowed date.
        end: Latest allowed date.

    Returns:
        A `date` object uniformly sampled between start and end.
    """
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def mk_id(row: Dict[str, str]) -> str:
    """Create a stable, short transaction ID from row fields using MD5.

    Why:
        We need an ID that (a) looks unique per transaction and (b) is reproducible
        across runs if the same content is generated again.

    Args:
        row: Dict with keys including date, description, merchant, amount, mode.

    Returns:
        A 10-character hex string derived from an MD5 hash.
    """
    raw = f"{row['date']}-{row['description']}-{row['merchant']}-{row['amount']}-{row['mode']}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def gen_row(category: str) -> Dict[str, str]:
    """Generate one synthetic transaction for a given category.

    Behavior:
        - Picks a random merchant and a plausible phrase for that category.
        - Injects variety (numbers, small “hinglish” tokens) to simulate mess.
        - Samples an amount from a category-specific range.
        - Randomizes payment mode and date.

    Args:
        category: One of the predefined CATEGORIES.

    Returns:
        A row dictionary matching the data contract, including a generated `id`.
    """
    merchant, phrases = random.choice(MERCHANTS[category])
    phrase = random.choice(phrases)

    # Add a random numeric suffix when the template contains "{}" (e.g., "uber*trip #{}")
    if "{}" in phrase:
        phrase = phrase.format(random.randint(1000, 9999))

    # Inject mild natural noise occasionally (e.g., "order", "bill", "ka", "txn")
    spice = random.choice([None, "ka", "bill", "order", "txn"])
    if spice and random.random() < 0.25:
        phrase = f"{phrase} {spice}"

    # Category-specific amount bands (rough but realistic)
    ranges = {
        "Food": (100, 800),
        "Transport": (150, 1200),
        "Bills": (300, 3000),
        "Groceries": (300, 2500),
        "Shopping": (200, 7000),
        "Rent": (8000, 25000),
        "Health": (200, 5000),
        "Insurance": (500, 20000),
        "Travel": (400, 20000),
        "Entertainment": (99, 1500),
    }
    lo, hi = ranges[category]
    amount = round(random.uniform(lo, hi), 2)

    mode = random.choice(MODES)
    d = random_date().strftime("%Y-%m-%d")

    row = {
        "id": "",  # filled below
        "date": d,
        "description": phrase.lower(),  # normalize casing like many real logs
        "merchant": merchant.lower(),
        "amount": f"{amount:.2f}",
        "mode": mode,
        "label": category,
    }
    row["id"] = mk_id(row)
    return row


def generate(n_per_class: int = 20) -> List[Dict[str, str]]:
    """Generate a balanced list of synthetic transactions.

    Why:
        A balanced dataset across categories makes it easy to sanity-check
        the training loop and baseline performance before we ingest real data.

    Args:
        n_per_class: Number of rows to produce per category.

    Returns:
        A shuffled list of row dicts (length = n_per_class * len(CATEGORIES)).
    """
    rows: List[Dict[str, str]] = []
    for cat in CATEGORIES:
        for _ in range(n_per_class):
            rows.append(gen_row(cat))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    """
    CLI entrypoint.

    Usage:
        python scripts/generate_synthetic_transactions.py
    Side effects:
        - Ensures a `data/` folder exists.
        - Writes `data/transactions.csv` with balanced synthetic rows.
    """
    import os

    os.makedirs("data", exist_ok=True)
    rows = generate(n_per_class=15)  # 10 categories * 15 = 150 rows
    with open("data/transactions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "date", "description", "merchant", "amount", "mode", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote data/transactions.csv with", len(rows), "rows")
