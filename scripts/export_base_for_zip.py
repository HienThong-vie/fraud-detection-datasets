"""Export the exact sampled Base model rows as plain CSV plus a compact data dictionary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

MEANINGS = {
    "_split": "Temporal partition used by the experiment: train, validation, or test.",
    "fraud_bool": "Target label: 1 fraudulent application, 0 legitimate application.",
    "income": "Applicant annual income in decile form.",
    "name_email_similarity": "Similarity between the applicant name and email; higher is more similar.",
    "prev_address_months_count": "Months at the previous registered address; missing when unavailable.",
    "current_address_months_count": "Months at the current registered address; missing when unavailable.",
    "customer_age": "Applicant age in years, rounded to the decade.",
    "days_since_request": "Days elapsed since the application was made.",
    "intended_balcon_amount": "Initial transfer amount for the application; negative source values are missing.",
    "payment_type": "Anonymized credit payment plan type.",
    "zip_count_4w": "Applications from the same ZIP code in the previous four weeks.",
    "velocity_6h": "Average applications per hour during the previous six hours.",
    "velocity_24h": "Average applications per hour during the previous 24 hours.",
    "velocity_4w": "Average applications per hour during the previous four weeks.",
    "bank_branch_count_8w": "Applications at the selected branch during the previous eight weeks.",
    "date_of_birth_distinct_emails_4w": "Distinct emails sharing the date of birth in four weeks.",
    "employment_status": "Anonymized applicant employment status.",
    "credit_risk_score": "Internal application credit-risk score.",
    "email_is_free": "Whether the applicant uses a free email domain.",
    "housing_status": "Anonymized applicant residential status.",
    "phone_home_valid": "Whether the supplied home phone is valid.",
    "phone_mobile_valid": "Whether the supplied mobile phone is valid.",
    "bank_months_count": "Age in months of a previous bank account; missing when unavailable.",
    "has_other_cards": "Whether the applicant has other cards from the same bank.",
    "proposed_credit_limit": "Credit limit requested by the applicant.",
    "foreign_request": "Whether the request country differs from the bank country.",
    "source": "Application channel: browser or telephone app.",
    "session_length_in_minutes": "Length of the application web session in minutes.",
    "device_os": "Operating system of the device used for the request.",
    "keep_alive_session": "User keep-alive choice at session logout.",
    "device_distinct_emails_8w": "Distinct emails seen from the device in the previous eight weeks.",
    "month": "Dataset month in the range 0 to 7, used for the temporal split.",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(source: Path, output: Path, dictionary: Path, manifest: Path, artifact: Path) -> dict:
    if not source.is_file():
        raise SystemExit(f"Run the real training workflow first; missing {source}")
    if not artifact.is_file():
        raise SystemExit(f"Missing Base champion artifact: {artifact}")

    output.parent.mkdir(parents=True, exist_ok=True)
    first = True
    rows = 0
    first_chunk = None
    for chunk in pd.read_csv(source, chunksize=50_000):
        if first_chunk is None:
            first_chunk = chunk
        chunk.to_csv(output, mode="w" if first else "a", header=first, index=False)
        rows += len(chunk)
        first = False
    if first_chunk is None:
        raise SystemExit(f"No rows found in {source}")

    model_inputs = set(joblib.load(artifact)["feature_columns"])
    data_dictionary = pd.DataFrame([
        {
            "column": column,
            "type": str(first_chunk[column].dtype),
            "meaning": MEANINGS.get(column, "See the official BAF datasheet."),
            "model_input": "yes" if column in model_inputs else "no",
        }
        for column in first_chunk.columns
    ])
    dictionary.parent.mkdir(parents=True, exist_ok=True)
    data_dictionary.to_csv(dictionary, index=False)

    payload = {
        "file": output.name,
        "rows": rows,
        "columns": len(first_chunk.columns),
        "size_bytes": output.stat().st_size,
        "size_mib": round(output.stat().st_size / 1024**2, 2),
        "sha256": file_sha256(output),
        "source": str(source),
        "dictionary": dictionary.name,
        "licence": "CC BY-NC-SA 4.0; synthetic BAF data; coursework/non-commercial use.",
        "note": "Exact sampled rows used by the final Base experiment after missing-value conversion.",
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/base_model_data.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("dataset/Base_final_model_data.csv"))
    parser.add_argument(
        "--dictionary", type=Path, default=Path("dataset/Base_final_model_data_dictionary.csv")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("dataset/Base_final_model_data_manifest.json")
    )
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/base/champion.joblib"))
    args = parser.parse_args()
    export(args.source, args.output, args.dictionary, args.manifest, args.artifact)


if __name__ == "__main__":
    main()
