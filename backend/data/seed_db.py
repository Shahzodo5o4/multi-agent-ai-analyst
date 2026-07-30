"""Build the demo database for the data agent (F5).

Creates `company.db` — a small but realistic SaaS dataset with a deliberate
story in it: churn spikes in Q3 2024, and the reasons for that spike are written
up in the documents the retriever searches. That's what makes the multi-agent
demo question work:

    "How many customers churned in Q3 2024, and why?"

The number can ONLY come from SQL. The reasons can ONLY come from the documents.
No single agent can answer it — which is the whole point of the project.

Run:  python data/seed_db.py
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "company.db"
random.seed(42)  # deterministic — everyone's database matches the guide's numbers

SCHEMA = """
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS churn_events;
DROP TABLE IF EXISTS support_tickets;

CREATE TABLE customers (
    id           INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    country      TEXT    NOT NULL,
    segment      TEXT    NOT NULL,   -- SMB | Mid-Market | Enterprise
    signup_date  TEXT    NOT NULL,
    status       TEXT    NOT NULL    -- active | churned
);

CREATE TABLE subscriptions (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    plan         TEXT    NOT NULL,   -- Starter | Growth | Scale
    mrr          REAL    NOT NULL,   -- monthly recurring revenue, USD
    start_date   TEXT    NOT NULL,
    end_date     TEXT               -- NULL while active
);

CREATE TABLE churn_events (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    churn_date   TEXT    NOT NULL,
    reason       TEXT    NOT NULL,
    mrr_lost     REAL    NOT NULL
);

CREATE TABLE support_tickets (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    opened_date  TEXT    NOT NULL,
    category     TEXT    NOT NULL,
    resolved     INTEGER NOT NULL   -- 0 | 1
);
"""

COUNTRIES = ["United States", "Germany", "United Kingdom", "France", "Canada",
             "Netherlands", "Spain", "Australia", "Sweden", "Poland"]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
PLANS = {"Starter": 49.0, "Growth": 199.0, "Scale": 749.0}
TICKET_CATEGORIES = ["billing", "onboarding", "performance", "integration",
                     "bug", "feature request"]

# Q3 2024 is the spike quarter. "pricing change" dominates it, matching the
# postmortem in data/docs/ — that link is what the critic verifies.
CHURN_REASONS_NORMAL = ["budget cut", "switched to competitor", "no longer needed",
                        "poor onboarding", "missing integration"]
CHURN_REASONS_Q3 = ["pricing change", "pricing change", "pricing change",
                    "switched to competitor", "missing integration", "budget cut"]


def daterange(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))


def build() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    customers, subscriptions, churns, tickets = [], [], [], []
    q3_start, q3_end = date(2024, 7, 1), date(2024, 9, 30)

    for cid in range(1, 241):  # 240 customers
        segment = random.choices(SEGMENTS, weights=[0.55, 0.30, 0.15])[0]
        plan = {"SMB": "Starter", "Mid-Market": "Growth", "Enterprise": "Scale"}[segment]
        signup = daterange(date(2022, 1, 1), date(2024, 6, 30))

        # ~20% churn overall, concentrated in Q3 2024.
        churned = random.random() < 0.20
        status = "churned" if churned else "active"
        customers.append((cid, f"Customer {cid:03d}",
                          random.choice(COUNTRIES), segment,
                          signup.isoformat(), status))

        mrr = PLANS[plan]
        end = None
        if churned:
            # 60% of all churn lands inside Q3 2024 — the spike.
            if random.random() < 0.60:
                churn_date = daterange(max(q3_start, signup + timedelta(days=30)), q3_end)
                reason = random.choice(CHURN_REASONS_Q3)
            else:
                churn_date = daterange(max(date(2023, 1, 1), signup + timedelta(days=30)),
                                       date(2025, 3, 31))
                reason = random.choice(CHURN_REASONS_NORMAL)
            end = churn_date.isoformat()
            churns.append((len(churns) + 1, cid, end, reason, mrr))

        subscriptions.append((cid, cid, plan, mrr, signup.isoformat(), end))

        for _ in range(random.randint(0, 4)):
            opened = daterange(signup, date(2025, 3, 31))
            tickets.append((len(tickets) + 1, cid, opened.isoformat(),
                            random.choice(TICKET_CATEGORIES),
                            1 if random.random() < 0.82 else 0))

    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)
    conn.executemany("INSERT INTO subscriptions VALUES (?,?,?,?,?,?)", subscriptions)
    conn.executemany("INSERT INTO churn_events VALUES (?,?,?,?,?)", churns)
    conn.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?)", tickets)
    conn.commit()

    q3_count = conn.execute(
        "SELECT COUNT(*) FROM churn_events WHERE churn_date BETWEEN '2024-07-01' AND '2024-09-30'"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    conn.close()

    print(f"Created {DB_PATH}")
    print(f"  customers        : {len(customers)}")
    print(f"  subscriptions    : {len(subscriptions)}")
    print(f"  churn_events     : {len(churns)}")
    print(f"  support_tickets  : {len(tickets)}")
    print(f"\n  Ground truth for your demo question:")
    print(f"    churned in Q3 2024        = {q3_count}")
    print(f"    as % of {total} customers    = {100 * q3_count / total:.1f}%")


if __name__ == "__main__":
    build()
