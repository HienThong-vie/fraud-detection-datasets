"""Dataset schema and user-facing notices."""

EXPECTED_BAF_COLUMNS = {
    "fraud_bool",
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "payment_type",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "credit_risk_score",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "source",
    "session_length_in_minutes",
    "device_os",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
    "month",
}

# These fields use documented negative missing-value sentinels. All use exactly
# -1 except intended_balcon_amount, for which every negative value means missing.
# Other numeric fields, such as credit_risk_score, may contain legitimate negatives.
MISSING_SENTINEL_COLUMNS = {
    "prev_address_months_count",
    "current_address_months_count",
    "intended_balcon_amount",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
}

ALL_NEGATIVE_MISSING_COLUMNS = {"intended_balcon_amount"}

BAF_VARIANT_PURPOSES = {
    "Base": "Reference dataset with no sampling bias intentionally introduced.",
    "Variant I": "Group-size disparity stress test.",
    "Variant II": "Fraud-prevalence disparity stress test.",
    "Variant III": "Group-dependent class-separability stress test.",
    "Variant IV": "Temporal prevalence-disparity stress test.",
    "Variant V": "Temporal class-separability stress test.",
}

SYNTHETIC_DATA_NOTICE = (
    "Educational decision-support prototype trained on synthetic BAF data. "
    "It must not approve, reject, or block a real bank application."
)
