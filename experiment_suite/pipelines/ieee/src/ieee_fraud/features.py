TRANSACTION_ID = "TransactionID"
TARGET = "isFraud"
TIME_COLUMN = "TransactionDT"

TRANSACTION_NUMERIC = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
]

TRANSACTION_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *[f"M{i}" for i in range(1, 10)],
]

IDENTITY_NUMERIC = [
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_11",
    "id_13",
    "id_14",
    "id_17",
    "id_19",
    "id_20",
    "id_32",
]

IDENTITY_CATEGORICAL = [
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
]

DERIVED_NUMERIC = ["TransactionDay", "TransactionHour", "TransactionWeekday"]
NUMERIC_FEATURES = TRANSACTION_NUMERIC + IDENTITY_NUMERIC + DERIVED_NUMERIC
CATEGORICAL_FEATURES = TRANSACTION_CATEGORICAL + IDENTITY_CATEGORICAL
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TRAIN_TRANSACTION_COLUMNS = [
    TRANSACTION_ID,
    TARGET,
    TIME_COLUMN,
    *TRANSACTION_NUMERIC,
    *TRANSACTION_CATEGORICAL,
]
TEST_TRANSACTION_COLUMNS = [
    TRANSACTION_ID,
    TIME_COLUMN,
    *TRANSACTION_NUMERIC,
    *TRANSACTION_CATEGORICAL,
]

