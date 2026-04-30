import pandas as pd
import pytest
from analyze_fraud import summarize_results


def make_scored(*rows):
    """Build a scored transactions DataFrame from a list of dicts."""
    return pd.DataFrame(rows)


def make_chargebacks(*transaction_ids):
    return pd.DataFrame({"transaction_id": list(transaction_ids)})


# ---------------------------------------------------------------------------
# Transaction counts per risk label
# ---------------------------------------------------------------------------

def test_transaction_count_per_label():
    scored = make_scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 200},
        {"transaction_id": 3, "risk_label": "low", "amount_usd": 50},
    )
    summary = summarize_results(scored, make_chargebacks())
    counts = dict(zip(summary["risk_label"], summary["transactions"]))
    assert counts["high"] == 2
    assert counts["low"] == 1


# ---------------------------------------------------------------------------
# Amount aggregations
# ---------------------------------------------------------------------------

def test_total_and_avg_amount_per_label():
    scored = make_scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 300},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 700},
        {"transaction_id": 3, "risk_label": "low",  "amount_usd": 50},
    )
    summary = summarize_results(scored, make_chargebacks())
    row = summary[summary["risk_label"] == "high"].iloc[0]
    assert row["total_amount_usd"] == pytest.approx(1000.0)
    assert row["avg_amount_usd"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Chargeback rate calculation
# ---------------------------------------------------------------------------

def test_chargeback_rate_is_zero_when_no_chargebacks():
    scored = make_scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 200},
    )
    summary = summarize_results(scored, make_chargebacks())
    row = summary[summary["risk_label"] == "high"].iloc[0]
    assert row["chargeback_rate"] == pytest.approx(0.0)


def test_chargeback_rate_is_one_when_all_chargebacks():
    scored = make_scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 200},
    )
    summary = summarize_results(scored, make_chargebacks(1, 2))
    row = summary[summary["risk_label"] == "high"].iloc[0]
    assert row["chargeback_rate"] == pytest.approx(1.0)


def test_chargeback_rate_partial():
    scored = make_scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 200},
        {"transaction_id": 3, "risk_label": "high", "amount_usd": 300},
        {"transaction_id": 4, "risk_label": "high", "amount_usd": 400},
    )
    summary = summarize_results(scored, make_chargebacks(1, 3))
    row = summary[summary["risk_label"] == "high"].iloc[0]
    assert row["chargeback_rate"] == pytest.approx(0.5)


def test_chargeback_rate_independent_per_label():
    """Chargebacks in one label must not pollute the rate of another."""
    scored = make_scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "low",  "amount_usd": 50},
    )
    summary = summarize_results(scored, make_chargebacks(1))
    high_row = summary[summary["risk_label"] == "high"].iloc[0]
    low_row  = summary[summary["risk_label"] == "low"].iloc[0]
    assert high_row["chargeback_rate"] == pytest.approx(1.0)
    assert low_row["chargeback_rate"] == pytest.approx(0.0)


def test_chargeback_count_matches_known_fraud():
    scored = make_scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 200},
        {"transaction_id": 3, "risk_label": "high", "amount_usd": 300},
    )
    summary = summarize_results(scored, make_chargebacks(1, 3))
    row = summary[summary["risk_label"] == "high"].iloc[0]
    assert row["chargebacks"] == 2
