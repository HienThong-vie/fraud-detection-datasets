"""Generate BAF-shaped data for local smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_frame(rows: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    month = np.arange(rows) % 8
    rng.shuffle(month)
    frame = pd.DataFrame(
        {
            "income": rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], rows),
            "name_email_similarity": rng.beta(2.2, 1.8, rows),
            "prev_address_months_count": rng.integers(-1, 180, rows),
            "current_address_months_count": rng.integers(-1, 240, rows),
            "customer_age": rng.choice([20, 30, 40, 50, 60, 70], rows),
            "days_since_request": rng.exponential(8, rows),
            "intended_balcon_amount": rng.normal(20, 45, rows),
            "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"], rows),
            "zip_count_4w": rng.integers(1, 2500, rows),
            "velocity_6h": rng.normal(3200, 1100, rows).clip(0),
            "velocity_24h": rng.normal(5000, 1600, rows).clip(0),
            "velocity_4w": rng.normal(4500, 900, rows).clip(0),
            "bank_branch_count_8w": rng.integers(0, 40, rows),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 12, rows),
            "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE", "CF"], rows),
            "credit_risk_score": rng.normal(100, 130, rows),
            "email_is_free": rng.integers(0, 2, rows),
            "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE"], rows),
            "phone_home_valid": rng.integers(0, 2, rows),
            "phone_mobile_valid": rng.integers(0, 2, rows),
            "bank_months_count": rng.integers(-1, 36, rows),
            "has_other_cards": rng.integers(0, 2, rows),
            "proposed_credit_limit": rng.choice([200, 500, 1000, 1500, 2000], rows),
            "foreign_request": rng.binomial(1, 0.04 + shift, rows),
            "source": rng.choice(["INTERNET", "TELEAPP"], rows, p=[0.92, 0.08]),
            "session_length_in_minutes": rng.exponential(8, rows),
            "device_os": rng.choice(["windows", "macintosh", "linux", "x11", "other"], rows),
            "keep_alive_session": rng.integers(0, 2, rows),
            "device_distinct_emails_8w": rng.integers(-1, 5, rows),
            "device_fraud_count": np.zeros(rows, dtype=int),
            "month": month,
        }
    )
    logit = (
        -4.9
        + 1.4 * frame["foreign_request"]
        + 0.9 * (frame["name_email_similarity"] < 0.2)
        + 0.7 * (frame["credit_risk_score"] < 0)
        + 0.6 * (frame["device_os"] == "other")
        + 0.35 * (frame["velocity_6h"] > 4300)
        + shift * 3
    )
    probability = 1 / (1 + np.exp(-logit))
    frame.insert(0, "fraud_bool", rng.binomial(1, probability))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/demo"))
    parser.add_argument("--rows", type=int, default=6000)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for index, (name, shift) in enumerate(
        [
            ("Base.csv", 0.0),
            ("Variant I.csv", 0.015),
            ("Variant II.csv", 0.03),
            ("Variant III.csv", 0.045),
            ("Variant IV.csv", 0.06),
            ("Variant V.csv", 0.075),
        ]
    ):
        make_frame(args.rows, 42 + index, shift).to_csv(args.destination / name, index=False)
    print(f"Demo-only data written to {args.destination.resolve()}")


if __name__ == "__main__":
    main()
