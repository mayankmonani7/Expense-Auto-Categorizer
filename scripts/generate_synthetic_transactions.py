import csv, random, hashlib
from datetime import date, timedelta

random.seed(42)

CATEGORIES = [
    "Food", "Transport", "Bills", "Groceries", "Shopping",
    "Rent", "Health", "Insurance", "Travel", "Entertainment"
]

MODES = ["UPI","Card","NetBanking","NEFT","Cash","Other"]

MERCHANTS = {
    "Food": [
        ("Zomato", ["zomato order", "zomato delivery", "zomato pizza deal"]),
        ("Swiggy", ["swiggy biryani", "swiggy lunch", "swiggy dinner"]),
        ("Starbucks", ["starbucks cappuccino", "starbucks latte", "starbucks cold brew"]),
        ("Dominos", ["dominos pizza", "dominos order", "dominos combo"])
    ],
    "Transport": [
        ("Uber", ["uber ride to office", "uber*trip #{}", "uber airport ride"]),
        ("Ola", ["ola cab city ride", "ola outstation", "ola*trip {}"]),
        ("Metro", ["metro smart card topup", "metro ticket", "metro pass renewal"]),
        ("IOCL", ["petrol pump iocl", "fuel iocl", "iocl fuel refill"])
    ],
    "Bills": [
        ("TNEB", ["electricity tneb bill", "tneb online payment"]),
        ("BSNL", ["bsnl broadband bill", "bsnl fiber bill"]),
        ("Airtel", ["airtel postpaid bill", "airtel dth recharge"]),
        ("TWAD", ["water board bill", "twad water payment"])
    ],
    "Groceries": [
        ("BigBasket", ["bigbasket veggies", "bigbasket grocery basket"]),
        ("Reliance Fresh", ["reliance fresh groceries", "reliance fresh milk & eggs"]),
        ("More", ["more supermarket staples", "more grocery bag"]),
        ("JioMart", ["jiomart monthly groceries", "jiomart dal & rice"])
    ],
    "Shopping": [
        ("Amazon", ["amazon basics hdmi cable", "amazon t-shirt", "amazon shoes"]),
        ("Flipkart", ["flipkart tshirt", "flipkart headphones", "flipkart home decor"]),
        ("Myntra", ["myntra kurti", "myntra jeans", "myntra footwear"]),
        ("Croma", ["croma usb-c charger", "croma power bank"])
    ],
    "Rent": [
        ("Bank Transfer", ["rent to landlord", "monthly house rent"]),
        ("GPay", ["rent via gpay", "gpay rent transfer"])
    ],
    "Health": [
        ("Apollo Pharmacy", ["apollo pharmacy medicine", "apollo antibiotic"]),
        ("Healthkart", ["healthkart whey protein", "healthkart vitamins"]),
        ("Dr Lal PathLabs", ["diagnostics blood test", "pathlab full checkup"]),
        ("Fortis", ["hospital opd fees", "fortis consultation"])
    ],
    "Insurance": [
        ("LIC", ["lic premium", "lic quarterly premium"]),
        ("HDFC Ergo", ["hdfc ergo motor insurance", "hdfc ergo health"]),
        ("ICICI Lombard", ["icici lombard renewal", "lombard policy premium"])
    ],
    "Travel": [
        ("IRCTC", ["irctc train booking", "irctc tatkal ticket"]),
        ("IndiGo", ["indigo flight add-on", "indigo web check-in"]),
        ("Goibibo", ["goibibo hotel booking", "goibibo bus ticket"]),
        ("OYO", ["oyo hotel stay", "oyo room booking"])
    ],
    "Entertainment": [
        ("Netflix", ["netflix monthly", "netflix subscription"]),
        ("Spotify", ["spotify subscription", "spotify family plan"]),
        ("BookMyShow", ["movie tickets bookmyshow", "bms weekend show"]),
        ("SonyLIV", ["sonyliv annual", "sonyliv quarterly"])
    ]
}

def random_date(start=date(2025, 7, 1), end=date(2025, 8, 10)):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def mk_id(row):
    raw = f"{row['date']}-{row['description']}-{row['merchant']}-{row['amount']}-{row['mode']}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

def gen_row(category):
    merchant, phrases = random.choice(MERCHANTS[category])
    phrase = random.choice(phrases)
    # inject minor variety
    if "{}" in phrase:
        phrase = phrase.format(random.randint(1000, 9999))
    # Hinglish/mixed cases (a bit of real-world mess)
    spice = random.choice([None, "ka", "bill", "order", "txn"])
    if spice and random.random() < 0.25:
        phrase = f"{phrase} {spice}"
    # amount ranges per category
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
        "Entertainment": (99, 1500)
    }
    lo, hi = ranges[category]
    amount = round(random.uniform(lo, hi), 2)
    mode = random.choice(MODES)
    d = random_date().strftime("%Y-%m-%d")
    row = {
        "id": "",
        "date": d,
        "description": phrase.lower(),
        "merchant": merchant.lower(),
        "amount": f"{amount:.2f}",
        "mode": mode,
        "label": category
    }
    row["id"] = mk_id(row)
    return row

def generate(n_per_class=20):
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_class):
            rows.append(gen_row(cat))
    # light shuffle but keep determinism
    random.shuffle(rows)
    return rows

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    rows = generate(n_per_class=15)  # 10 classes * 15 = 150 rows
    with open("data/transactions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","date","description","merchant","amount","mode","label"])
        w.writeheader()
        w.writerows(rows)
    print("Wrote data/transactions.csv with", len(rows), "rows")