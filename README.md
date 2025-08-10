# Expense-Auto-Categorizer

Automatically classify personal transactions (bank SMS/UPI/CSV) into categories such as **Food, Transport, Bills, Groceries, Shopping, Rent, Health, Insurance, Travel, Entertainment**.

## Goal
Turn messy real-world descriptions (e.g., “Uber*TRIP 872”, “Amazon Fresh veggies”) into clean categories for budgeting—then keep the model healthy in production with proper MLOps.

## Acceptance Criteria (v1)
- **Macro-F1 ≥ 0.80** on a 2-week **time-based holdout**
- **p95 latency < 120 ms** per request on local Docker
- **Drift alert** if weekly class distribution or text embedding distance exceeds **3σ** from trailing 30-day baseline
- **Safe rollback**: auto-switch to the previous model if live accuracy proxy drops >5% for 24h

## Categories (v1)
`Food, Transport, Bills, Groceries, Shopping, Rent, Health, Insurance, Travel, Entertainment`

## Data Contract (input schema)
| column       | type   | required | constraints/notes |
|--------------|--------|----------|-------------------|
| `id`         | string | yes      | unique within file |
| `date`       | string | yes      | ISO `YYYY-MM-DD`  |
| `description`| string | yes      | length 3–300      |
| `merchant`   | string | no       | default: empty    |
| `amount`     | float  | yes      | ≥ 0               |
| `mode`       | string | no       | one of {UPI, Card, NetBanking, NEFT, Cash, Other} |
| `label`      | string | training | one of the categories above (not required at inference) |

### Labeling Guide (short)
- **Food**: Swiggy/Zomato/Starbucks/Domino’s (groceries ≠ Food → **Groceries**)
- **Transport**: Uber/Ola/cabs/metro/fuel (if clearly travel → **Travel**)
- **Bills**: electricity/water/internet/phone
- **Rent**: monthly landlord transfers
- **Shopping**: non-grocery e-commerce (apparel/gadgets)
- **Health**: pharmacy/hospital/diagnostics
- **Insurance**: LIC/health/auto insurance
- **Travel**: flights/trains/hotels (IRCTC, airlines)
- **Entertainment**: OTT/events/gaming
- **Groceries**: BigBasket/Reliance Fresh/supermarkets

**Edge rules**  
- If merchant suggests multiple categories, **description wins** (“Amazon electricity bill” → **Bills**).  
- Drop rows with `amount == 0`.  
- Empty `description` → manual review bucket (don’t train on it).

## Roadmap (high level)
1. Data ingestion & validation (Great Expectations)
2. Baseline model (TF-IDF + Logistic Regression) with time-based split
3. Experiment tracking & model registry
4. FastAPI service for online inference
5. Docker + CI (tests, build)
6. Monitoring (Evidently for drift; Prometheus/Grafana for latency/errors)
7. Scheduled retraining + canary + rollback

## Repo Structure (to be created)

├─ data/
├─ notebooks/
├─ src/
├─ tests/
├─ monitoring/
├─ infra/
└─ README.md
