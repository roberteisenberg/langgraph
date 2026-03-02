"""
Canonical test cases — same 6 inputs across all phases.

Each phase runs these and demonstrates progressively better handling.
"""

CASE_1_OBVIOUSLY_LEGIT = {
    "id": "order-001",
    "customer_email": "alice.johnson@gmail.com",
    "customer_name": "Alice Johnson",
    "amount": 45.00,
    "items": [
        {"name": "Wireless Mouse", "quantity": 1, "price": 25.00},
        {"name": "Mouse Pad", "quantity": 1, "price": 20.00},
    ],
    "shipping_address": "742 Evergreen Terrace, Springfield, IL 62704",
    "metadata": {
        "account_age_days": 1825,  # 5 years
        "prior_orders": 50,
        "prior_fraud_flags": 0,
    },
}

CASE_2_MILDLY_SUSPICIOUS = {
    "id": "order-002",
    "customer_email": "newbuyer_2026@outlook.com",
    "customer_name": "Jordan Smith",
    "amount": 380.00,
    "items": [
        {"name": "Noise-Canceling Headphones", "quantity": 1, "price": 280.00},
        {"name": "Headphone Stand", "quantity": 1, "price": 50.00},
        {"name": "USB-C Cable", "quantity": 2, "price": 25.00},
    ],
    "shipping_address": "Apt 14B, 900 Lake Shore Dr, Chicago, IL 60611",
    "metadata": {
        "account_age_days": 12,
        "prior_orders": 0,
        "prior_fraud_flags": 0,
    },
}

CASE_3_HIGH_RISK = {
    "id": "order-003",
    "customer_email": "xuser8819@tempmail.ninja",
    "customer_name": "Chris Martinez",
    "amount": 2800.00,
    "items": [
        {"name": "MacBook Pro 16-inch", "quantity": 1, "price": 2500.00},
        {"name": "AppleCare+", "quantity": 1, "price": 300.00},
    ],
    "shipping_address": "Unit 7, 3100 Industrial Pkwy, Hayward, CA 94545",
    "metadata": {
        "account_age_days": 2,
        "prior_orders": 0,
        "prior_fraud_flags": 0,
    },
}

CASE_4_CONFLICTING_SIGNALS = {
    "id": "order-004",
    "customer_email": "robert.chen@company.com",
    "customer_name": "Robert Chen",
    "amount": 1500.00,
    "items": [
        {"name": "Standing Desk", "quantity": 1, "price": 800.00},
        {"name": "Monitor Arm", "quantity": 2, "price": 200.00},
        {"name": "Ergonomic Chair", "quantity": 1, "price": 300.00},
    ],
    "shipping_address": "Bay 12, 5500 Commerce Way, Reno, NV 89502",
    "metadata": {
        "account_age_days": 730,  # 2 years
        "prior_orders": 30,
        "prior_fraud_flags": 0,
        "recent_ip_change": True,
        "shipping_name_mismatch": True,  # shipping name != account name
        "address_shared_by_accounts": 3,  # same address used by 2 other accounts
    },
}

CASE_5_HISTORICAL_FRAUD = {
    "id": "order-005",
    "customer_email": "pat.williams@yahoo.com",
    "customer_name": "Pat Williams",
    "amount": 60.00,
    "items": [
        {"name": "Bluetooth Speaker", "quantity": 1, "price": 60.00},
    ],
    "shipping_address": "221B Baker Street, Portland, OR 97201",
    "metadata": {
        "account_age_days": 365,
        "prior_orders": 8,
        "prior_fraud_flags": 1,  # flagged 6 months ago
        "last_flag_days_ago": 180,
    },
}

CASE_6_TOOL_ERROR = {
    "id": "order-006",
    "customer_email": "sam.taylor@gmail.com",
    "customer_name": "Sam Taylor",
    "amount": 150.00,
    "items": [
        {"name": "Mechanical Keyboard", "quantity": 1, "price": 150.00},
    ],
    "shipping_address": "456 Oak Avenue, Austin, TX 78701",
    "metadata": {
        "account_age_days": 200,
        "prior_orders": 5,
        "prior_fraud_flags": 0,
        "simulate_payment_error": True,  # triggers tool error in later phases
    },
}

ALL_CASES = [
    ("Case 1: Obviously Legit", CASE_1_OBVIOUSLY_LEGIT),
    ("Case 2: Mildly Suspicious", CASE_2_MILDLY_SUSPICIOUS),
    ("Case 3: High Risk", CASE_3_HIGH_RISK),
    ("Case 4: Conflicting Signals", CASE_4_CONFLICTING_SIGNALS),
    ("Case 5: Historical Fraud", CASE_5_HISTORICAL_FRAUD),
    ("Case 6: Tool Error", CASE_6_TOOL_ERROR),
]
