import pandas as pd
import pytest
from features import build_model_frame


def make_transactions(**overrides):
    row = {
        "transaction_id": 1,
        "account_id": 100,
        "amount_usd": 50.0,
        "failed_logins_24h": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def make_accounts():
    return pd.DataFrame([{"account_id": 100, "prior_chargebacks": 0, "country": "US"}])


# ---------------------------------------------------------------------------
# is_large_amount
# ---------------------------------------------------------------------------

def test_is_large_amount_true_at_threshold():
    df = build_model_frame(make_transactions(amount_usd=1000), make_accounts())
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_true_above_threshold():
    df = build_model_frame(make_transactions(amount_usd=2500), make_accounts())
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_false_below_threshold():
    df = build_model_frame(make_transactions(amount_usd=999.99), make_accounts())
    assert df["is_large_amount"].iloc[0] == 0


# ---------------------------------------------------------------------------
# login_pressure
# ---------------------------------------------------------------------------

def test_login_pressure_none_at_zero():
    df = build_model_frame(make_transactions(failed_logins_24h=0), make_accounts())
    assert df["login_pressure"].iloc[0] == "none"


def test_login_pressure_low():
    for n in (1, 2):
        df = build_model_frame(make_transactions(failed_logins_24h=n), make_accounts())
        assert df["login_pressure"].iloc[0] == "low", f"Expected low for failed_logins_24h={n}"


def test_login_pressure_high():
    for n in (3, 10):
        df = build_model_frame(make_transactions(failed_logins_24h=n), make_accounts())
        assert df["login_pressure"].iloc[0] == "high", f"Expected high for failed_logins_24h={n}"


# ---------------------------------------------------------------------------
# merge behaviour
# ---------------------------------------------------------------------------

def test_account_columns_present_after_merge():
    df = build_model_frame(make_transactions(), make_accounts())
    assert "prior_chargebacks" in df.columns
    assert "country" in df.columns


def test_unmatched_account_produces_nulls():
    txns = make_transactions(account_id=999)
    df = build_model_frame(txns, make_accounts())
    assert pd.isna(df["prior_chargebacks"].iloc[0])
